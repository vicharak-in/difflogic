import torch
import math
from .difflogic import LogicLayer, GroupSum
import tempfile
import subprocess
import shutil
import ctypes
import numpy as np
import numpy.typing
import time
from typing import Union
import os

ALL_OPERATIONS = [
    "zero",
    "and",
    "not_implies",
    "a",
    "not_implied_by",
    "b",
    "xor",
    "or",
    "not_or",
    "not_xor",
    "not_b",
    "implied_by",
    "not_a",
    "implies",
    "not_and",
    "one",
]

BITS_TO_DTYPE = {8: "char", 16: "short", 32: "int", 64: "long long"}
BITS_TO_ZERO_LITERAL = {8: "(char) 0",
                        16: "(short) 0", 32: "0", 64: "0LL"}
BITS_TO_ONE_LITERAL = {8: "(char) 1",
                        16: "(short) 1", 32: "1", 64: "1LL"}
BITS_TO_C_DTYPE = {8: ctypes.c_int8, 16: ctypes.c_int16,
                   32: ctypes.c_int32, 64: ctypes.c_int64}
BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


class CompiledLogicNet(torch.nn.Module):
    def __init__(
            self,
            model: torch.nn.Sequential,
            device='cpu',
            num_bits=64,
            cpu_compiler='gcc',
            max_layer=None,
            verbose=False,
    ):
        super(CompiledLogicNet, self).__init__()
        self.model = model
        self.device = device
        self.num_bits = num_bits
        self.cpu_compiler = cpu_compiler
        self.max_layer = max_layer

        assert cpu_compiler in ["clang", "gcc"], cpu_compiler
        assert num_bits in [8, 16, 32, 64]

        if self.model is not None:
            layers = []

            self.num_inputs = None
            self.num_outputs = None

            assert isinstance(self.model[-1], GroupSum), 'The last layer of the model must be GroupSum, but it is {} / {}' \
                                                         ' instead.'.format(type(self.model[-1]), self.model[-1])
            self.num_classes = self.model[-1].k

            first = True
            for layer in self.model:
                if isinstance(layer, LogicLayer):
                    if first:
                        self.num_inputs = layer.in_dim
                        first = False
                    self.num_outputs = layer.out_dim
                    self.num_out_per_class = self.num_outputs // self.num_classes
                    layers.append((layer.indices[0], layer.indices[1], layer.weights.argmax(1)))
                elif isinstance(layer, torch.nn.Flatten):
                    if verbose:
                        print('Skipping torch.nn.Flatten layer ({}).'.format(type(layer)))
                elif isinstance(layer, GroupSum):
                    if verbose:
                        print('Skipping GroupSum layer ({}).'.format(type(layer)))
                else:
                    assert False, 'Error: layer {} / {} unknown.'.format(type(layer), layer)

            self.layers = layers

            if verbose:
                print('`layers` created and has {} layers.'.format(len(layers)))

        self.lib_fn = None

    def get_gate_code(self, var1, var2, gate_op):
        operation_name = ALL_OPERATIONS[gate_op]

        if operation_name == "zero":
            res = BITS_TO_ZERO_LITERAL[self.num_bits]
        elif operation_name == "and":
            res = f"{var1} & {var2}"
        elif operation_name == "not_implies":
            res = f"{var1} & ~{var2}"
        elif operation_name == "a":
            res = f"{var1}"
        elif operation_name == "not_implied_by":
            res = f"{var2} & ~{var1}"
        elif operation_name == "b":
            res = f"{var2}"
        elif operation_name == "xor":
            res = f"{var1} ^ {var2}"
        elif operation_name == "or":
            res = f"{var1} | {var2}"
        elif operation_name == "not_or":
            res = f"~({var1} | {var2})"
        elif operation_name == "not_xor":
            res = f"~({var1} ^ {var2})"
        elif operation_name == "not_b":
            res = f"~{var2}"
        elif operation_name == "implied_by":
            res = f"~{var2} | {var1}"
        elif operation_name == "not_a":
            res = f"~{var1}"
        elif operation_name == "implies":
            res = f"~{var1} | {var2}"
        elif operation_name == "not_and":
            res = f"~({var1} & {var2})"
        elif operation_name == "one":
            res = f"~{BITS_TO_ZERO_LITERAL[self.num_bits]}"
        else:
            assert False, 'Operator {} unknown.'.format(operation_name)

        if self.num_bits == 8:
            res = f"(char) ({res})"
        elif self.num_bits == 16:
            res = f"(short) ({res})"

        return res

    def get_layer_code(self, layer_a, layer_b, layer_op, layer_id, prefix_sums):
        code = []
        for var_id, (gate_a, gate_b, gate_op) in enumerate(zip(layer_a, layer_b, layer_op)):
            if self.device == 'cpu' and layer_id == len(prefix_sums) - 1:
                a = f"v{prefix_sums[layer_id - 1] + gate_a}"
                b = f"v{prefix_sums[layer_id - 1] + gate_b}"
                code.append(f"\tout[{var_id}] = {self.get_gate_code(a, b, gate_op)};")
            else:
                assert not (self.device == 'cpu' and layer_id >= len(prefix_sums) - 1), (layer_id, len(prefix_sums))
                if layer_id == 0:
                    a = f"inp[{gate_a}]"
                    b = f"inp[{gate_b}]"
                else:
                    a = f"v{prefix_sums[layer_id - 1] + gate_a}"
                    b = f"v{prefix_sums[layer_id - 1] + gate_b}"
                code.append(
                    f"\tconst {BITS_TO_DTYPE[self.num_bits]} v{prefix_sums[layer_id] + var_id} = {self.get_gate_code(a, b, gate_op)};"
                )
        return code

    def get_c_code(self):
        prefix_sums = [0]
        cur_count = 0
        for layer_a, layer_b, layer_op in self.layers[:(self.max_layer or len(self.layers))-1]:
            cur_count += len(layer_a)
            prefix_sums.append(cur_count)

        code = [
            "#include <stddef.h>",
            "#include <stdlib.h>",
            "#include <stdbool.h>",
            "#include <stdio.h>",
            "#include <inttypes.h>  // for PRIu64, etc",
            "",
            f"void logic_gate_net({BITS_TO_DTYPE[self.num_bits]} const *inp, {BITS_TO_DTYPE[self.num_bits]} *out) {{",
        ]

        for layer_id, (layer_a, layer_b, layer_op) in enumerate(self.layers[:self.max_layer]):
            code.extend(self.get_layer_code(layer_a, layer_b, layer_op, layer_id, prefix_sums))

        code.append("}")

        num_neurons_ll = self.layers[-1][0].shape[0]
        log2_of_num_neurons_per_class_ll = math.ceil(math.log2(num_neurons_ll / self.num_classes + 1))
        num_neurons_per_class = self.layers[-1][0].shape[0] // self.num_classes
        
        code.append(f"""
void apply_logic_gate_net(bool const *inp, {BITS_TO_DTYPE[32]} *out, size_t len) {{
    {BITS_TO_DTYPE[self.num_bits]} *inp_temp = malloc({self.num_inputs} * sizeof({BITS_TO_DTYPE[self.num_bits]}));
    {BITS_TO_DTYPE[self.num_bits]} *out_temp = malloc({num_neurons_ll} * sizeof({BITS_TO_DTYPE[self.num_bits]}));
    {BITS_TO_DTYPE[self.num_bits]} *out_temp_o = malloc({log2_of_num_neurons_per_class_ll} * sizeof({BITS_TO_DTYPE[self.num_bits]}));

    FILE *log_file = fopen("inference_log.txt", "w");
    if (!log_file) {{
        perror("Failed to open log file");
        return;
    }}

    for (size_t i = 0; i < len; ++i) {{

        // Print raw input bits
        fprintf(log_file, "Sample %zu raw input bits:\\n", i);
        for (size_t b = 0; b < {self.num_inputs} * {self.num_bits}; ++b) {{
            fprintf(log_file, "%d", inp[i * {self.num_inputs} * {self.num_bits} + b]);
            if ((b + 1) % {self.num_inputs} == 0) fprintf(log_file, "\\n");
        }}

        // Converting bool array to bitpacked inp_temp
        for (size_t d = 0; d < {self.num_inputs}; ++d) {{
            {BITS_TO_DTYPE[self.num_bits]} res = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            for (size_t b = 0; b < {self.num_bits}; ++b) {{
                res <<= 1;
                res += !!(inp[i * {self.num_inputs} * {self.num_bits} + ({self.num_bits} - b - 1) * {self.num_inputs} + d]);
            }}
            inp_temp[d] = res;
        }}

        // Print bitpacked inputs
        fprintf(log_file, "Sample %zu bitpacked inputs:\\n", i);
        for (size_t d = 0; d < {self.num_inputs}; ++d) {{
            fprintf(log_file, "inp_temp[%zu] = %" PRIu64 "\\n", d, (uint64_t)inp_temp[d]);
        }}

        // Apply logic gate network
        logic_gate_net(inp_temp, out_temp);

        for (size_t c = 0; c < {self.num_classes}; ++c) {{

            // Clear output temp for sum
            for (size_t d = 0; d < {log2_of_num_neurons_per_class_ll}; ++d) {{
                out_temp_o[d] = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            }}

            for (size_t a = 0; a < {self.layers[-1][0].shape[0] // self.num_classes}; ++a) {{
                {BITS_TO_DTYPE[self.num_bits]} carry = out_temp[c * {self.layers[-1][0].shape[0] // self.num_classes} + a];
                {BITS_TO_DTYPE[self.num_bits]} out_temp_o_d;
                for (int d = {log2_of_num_neurons_per_class_ll} - 1; d >= 0; --d) {{
                    out_temp_o_d = out_temp_o[d];
                    out_temp_o[d] = carry ^ out_temp_o_d;
                    carry = carry & out_temp_o_d;
                }}
            }}

            // Print result bits after group adder
            fprintf(log_file, "Class %zu - Output bits after adder (out_temp_o):\\n", c);
            for (size_t d = 0; d < {log2_of_num_neurons_per_class_ll}; ++d) {{
                fprintf(log_file, "out_temp_o[%zu] = %" PRIu64 "\\n", d, (uint64_t)out_temp_o[d]);
            }}

            // Final unpacked output bits
            for (size_t b = 0; b < {self.num_bits}; ++b) {{
                const {BITS_TO_DTYPE[self.num_bits]} bit_mask = {BITS_TO_ONE_LITERAL[self.num_bits]} << b;
                {BITS_TO_DTYPE[32]} res = 0;
                for (size_t d = 0; d < {log2_of_num_neurons_per_class_ll}; ++d) {{
                    res <<= 1;
                    res += !!(out_temp_o[d] & bit_mask);
                }}
                out[(i * {self.num_bits} + b) * {self.num_classes} + c] = res;

                // Log each final output result
                fprintf(log_file, "Output[%zu][bit %zu][class %zu] = %" PRIu32 "\\n", i, b, c, (uint32_t)res);
            }}
        }}
    }}

    // --- LOGGING TO in_out.csv STARTS HERE ---

    // Open CSV output file for appending results
    FILE* csv_file = fopen("in_out.csv", "a");
    if (!csv_file) {{
        perror("Failed to open in_out.csv");
        exit(EXIT_FAILURE);
    }}
    
    for (size_t i = 0; i < 1; ++i) {{
        // 1. Log input bits (400 bits)
        for (size_t b = 0; b < 400; ++b) {{
            size_t bit_index = i * 400 + b;
            size_t byte_index = bit_index / 8;
            size_t bit_offset = 7 - (bit_index % 8); // big endian
            uint8_t bit = (inp[byte_index] >> bit_offset) & 1;
            fprintf(csv_file, "%d", bit);
        }}

        // 2. Log output bits after adder for each class
        for (size_t c = 0; c < {self.num_classes}; ++c) {{
            for (size_t d = 0; d < {log2_of_num_neurons_per_class_ll}; ++d) {{
                fprintf(csv_file, ",%" PRIu64, (uint64_t)out_temp_o[c * {log2_of_num_neurons_per_class_ll} + d]);
            }}
        }}

        // 3. Compute and log final prediction
        uint32_t class_scores[{self.num_classes}];
        for (size_t c = 0; c < {self.num_classes}; ++c) {{
            uint32_t val = 0;
            for (size_t b = 0; b < {self.num_bits}; ++b) {{
                val <<= 1;
                val += out[(i * {self.num_bits} + b) * {self.num_classes} + c];
            }}
            class_scores[c] = val;
        }}

        size_t predicted_class = 0;
        uint32_t max_val = class_scores[0];
        for (size_t c = 1; c < {self.num_classes}; ++c) {{
            if (class_scores[c] > max_val) {{
                max_val = class_scores[c];
                predicted_class = c;
            }}
        }}

        fprintf(csv_file, ",%zu", predicted_class);
    }}

    fclose(csv_file);
    // --- LOGGING TO in_out.csv ENDS HERE ---

    fclose(log_file);
    free(inp_temp);
    free(out_temp);
    free(out_temp_o);
}}""")
        return "\n".join(code)

    def compile(self, opt_level=1, save_lib_path=None, verbose=False):
        """
        Regarding the optimization level for C compiler:

        compilation time vs. call time for 48k lines of code
        -O0 -> 5.5s compiling -> 269ms call
        -O1 -> 190s compiling -> 125ms call
        -O2 -> 256s compiling -> 130ms call
        -O3 -> 346s compiling -> 124ms call

        :param opt_level: optimization level for C compiler
        :param save_lib_path: (optional) where to save the .so shared object library
        :param verbose:
        :return:
        """

        with tempfile.NamedTemporaryFile(suffix=".so") as lib_file:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".c" if self.device != "cuda" else ".cu"
            ) as c_file:
                if self.device == 'cpu':
                    code = self.get_c_code()
                else:
                    assert False, 'Device {} not supported.'.format(self.device)

                if verbose and len(code.split('\n')) <= 200:
                    print()
                    print()
                    print(code)
                    print()
                    print()

                c_file.write(code)
                c_file.flush()

                name=f"compiled_c_{self.num_bits}bits_{self.num_outputs}.c"
                path='./saved_files'
                if not os.path.exists(path):
                    os.makedirs(path)
                local_c_file=os.path.join(path, name)
                shutil.copy(c_file.name, local_c_file)

                if verbose:
                    print('C code created and has {} lines. (temp location {})'.format(len(code.split('\n')), c_file.name))
                    print(f"C code is copied from {c_file.name} to {local_c_file}")

                t_s = time.time()
                if self.device == 'cpu':
                    compiler_out = subprocess.run(
                        [
                            self.cpu_compiler,
                            "-shared",
                            "-fPIC",
                            "-O{}".format(opt_level),
                            # "-march=native",  # removed for compatibility with Apple Silicon: https://stackoverflow.com/questions/65966969/why-does-march-native-not-work-on-apple-m1
                            "-o",
                            lib_file.name,
                            c_file.name,
                        ]
                    )
                else:
                    assert False, 'Device {} not supported.'.format(self.device)

                if compiler_out.returncode != 0:
                    raise RuntimeError(
                        f'compilation exited with error code {compiler_out.returncode}')

                print('Compiling finished in {:.3f} seconds.'.format(time.time() - t_s))

            if save_lib_path is not None:
                shutil.copy(lib_file.name, save_lib_path)
                if verbose:
                    print('C lib file named {} copied to {} successfully....'.format(lib_file.name, save_lib_path))

            lib = ctypes.cdll.LoadLibrary(lib_file.name)

            lib_fn = lib.apply_logic_gate_net
            lib_fn.restype = None
            lib_fn.argtypes = [
                np.ctypeslib.ndpointer(
                    ctypes.c_bool, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(
                    BITS_TO_C_DTYPE[32], flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
            ]

        self.lib_fn = lib_fn

    def get_verilog_gate_code(self, var1, var2, gate_op):
        operation_name = ALL_OPERATIONS[gate_op]

        if operation_name == "zero":
            res="1'b0"
        elif operation_name == "and":
            res=f"{var1} & {var2}"
        elif operation_name == "not_implies":
            res=f"{var1} & ~{var2}"
        elif operation_name == "a":
            res=f"{var1}"
        elif operation_name == "not_implied_by":
            res=f"{var2} & ~{var1}"
        elif operation_name == "b":
            res=f"{var2}"
        elif operation_name == "xor":
            res=f"{var1} ^ {var2}"
        elif operation_name == "or":
            res=f"{var1} | {var2}"
        elif operation_name == "not_or":
            res=f"~({var1} | {var2})"
        elif operation_name == "not_xor":
            res=f"~({var1} ^ {var2})"
        elif operation_name == "not_b":
            res=f"~{var2}"
        elif operation_name == "implied_by":
            res=f"~{var2} | {var1}"
        elif operation_name == "not_a":
            res=f"~{var1}"
        elif operation_name == "implies":
            res=f"~{var1} | {var2}"
        elif operation_name == "not_and":
            res=f"~({var1} & {var2})"
        elif operation_name == "one":
            res="1'b1"
        else:
            assert False, f"Operator {operation_name} unknown."
        return res

    def get_verilog_code(self):
        """Generates Verilog code for the logic network."""
        max_layer = self.max_layer if self.max_layer is not None else len(self.layers)
        neurons_per_class=self.num_outputs // self.num_classes
        log2_neurons_per_class=math.ceil(math.log2(neurons_per_class + 1))
        output_bits=self.num_classes * log2_neurons_per_class
        last_layer_output= []
        code = [
            "module logic_network ("
            f"    input clk,",
            f"    input wire [{self.num_inputs-1}:0] x,",
            f"    output wire [{output_bits-1}:0] y",
            ");",
        ]

        for layer_id in range(max_layer):
            print(f"{self.layers[layer_id]}: ", self.layers[layer_id])
            layer_size = len(self.layers[layer_id][0])  #no. of neurons in a layer

            code.append(f"      reg [{self.num_outputs-1}:0] layer{layer_id}_out = 0;")

        code.append(f"      reg [{self.num_outputs-1}:0] last_layer_output = 0;")
        code.append(f"      reg [{log2_neurons_per_class-1}:0] result [{self.num_classes-1}:0];")
        code.append(f"      always @(posedge clk) begin")

        for layer_id, (layer_a, layer_b, layer_op) in enumerate(self.layers[:self.max_layer]):
            layer_size = len(layer_a)
            for var_id, (gate_a, gate_b, gate_op) in enumerate(zip(layer_a, layer_b, layer_op)):
                if layer_id == 0:
                    a = f"x[{gate_a}]"
                    b = f"x[{gate_b}]"
                else:
                    a = f"layer{layer_id-1}_out[{gate_a}]"
                    b = f"layer{layer_id-1}_out[{gate_b}]"

                gate_expr = self.get_verilog_gate_code(a, b, gate_op)
                code.append(f"     layer{layer_id}_out[{var_id}] <= {gate_expr};")

            if layer_id==max_layer-1:
                for i in range(self.num_outputs):
                    last_layer_output.append(f"layer{layer_id}_out[{i}]")
        code.append(f"      last_layer_output <= layer{max_layer-1}_out;")
        code.append("")

        for cls in range(self.num_classes):
            base_index = cls * neurons_per_class
            terms = [f"last_layer_output[{base_index + n}]" for n in range(neurons_per_class)]
            joined = " + ".join(terms)
            code.append(f"      result[{cls}] <= {joined};")

        code.append("end")
        count=0
        for i in range(output_bits,0,-log2_neurons_per_class):
            code.append(f"      assign y[{i-1}:{i-log2_neurons_per_class}]=result[{count}];")
            count=count+1

        code.append("endmodule")
        return "\n".join(code)

    def compile_verilog(self, save_folder=None, verbose=True):
        if self.device != 'cpu':
            raise ValueError(f"Device {self.device} not supported.")

        code = self.get_verilog_code()

        if verbose and len(code.split('\n')) <= 200:
            print("\n\n" + code + "\n\n")

        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            name = f"compiled_verilog_{self.num_bits}_{self.num_outputs}.v"
            v_file_path = os.path.join(save_folder, name)
        else:
            # Use a temporary file if no folder is provided
            v_file = tempfile.NamedTemporaryFile(mode="w", suffix=".v", delete=False)
            v_file_path = v_file.name

        with open(v_file_path, "w") as v_file:
            v_file.write(code)

        if verbose:
            print(f"Verilog code saved at: {v_file_path}")
            print(f"Verilog code has {len(code.splitlines())} lines.")

    @staticmethod
    def load(save_lib_path, num_classes, num_bits):

        self = CompiledLogicNet(None, num_bits=num_bits)
        self.num_classes = num_classes

        lib = ctypes.cdll.LoadLibrary(save_lib_path)

        lib_fn = lib.apply_logic_gate_net
        lib_fn.restype = None
        lib_fn.argtypes = [
            np.ctypeslib.ndpointer(
                ctypes.c_bool, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(
                BITS_TO_C_DTYPE[32], flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
        ]

        self.lib_fn = lib_fn
        return self

    def forward(
            self,
            x: Union[torch.BoolTensor, numpy.typing.NDArray[np.bool_]],
            verbose: bool = False
    ) -> torch.IntTensor:
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        batch_size_div_bits = math.ceil(x.shape[0] / self.num_bits)
        pad_len = batch_size_div_bits * self.num_bits - x.shape[0]
        x = np.concatenate([x, np.zeros_like(x[:pad_len])])

        if verbose:
            print('x.shape', x.shape)

        out = np.zeros(
            x.shape[0] * self.num_classes, dtype=BITS_TO_NP_DTYPE[32]
        )
        x = x.reshape(-1)

        self.lib_fn(x, out, batch_size_div_bits)

        out = torch.tensor(out).view(batch_size_div_bits * self.num_bits, self.num_classes)
        if pad_len > 0:
            out = out[:-pad_len]
        if verbose:
            print('out.shape', out.shape)

        return out

