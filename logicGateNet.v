module logicGateNet #(
  parameter INPUT_BITS = 400,
  parameter NEURONS = 800,
  parameter CLASSES = 10,
  parameter GROUPS = NEURONS / CLASSES,
  parameter BITS_PER_VALUE = $clog2(GROUPS),
  parameter OUTPUT_BITS = BITS_PER_VALUE * CLASSES
)
  (
    input clk,
    input rx,
    output tx
  );

  wire [7:0] rx_data;
  wire rx_ready;
  reg [INPUT_BITS-1:0] input_bits = 0;
  wire [OUTPUT_BITS-1:0] output_bits;
  reg [7:0] tx_data = 0;
  reg send;

  uart_rx uart_rx_inst (
    .i_Clock(clk),
    .i_Rx_Serial(rx),
    .o_Rx_Byte(rx_data),
    .o_Rx_DV(rx_ready)
  );

  localparam INPUT_BYTES = INPUT_BITS / 8;
  localparam TX_ITERATIONS = OUTPUT_BITS / BITS_PER_VALUE;
  localparam ZERO_PADS = 8 - BITS_PER_VALUE;

  reg [15:0] tx_count = 0;
  reg [7:0] state = 0;
  reg [15:0] byte_count = 0;

  reg [7:0] infer_count = 0;

  always @(posedge clk) begin
    case(state)
      0: begin /* reading */
        if (rx_ready && byte_count < INPUT_BYTES) begin
          input_bits[byte_count * 8 +: 8] <= rx_data;
          byte_count <= byte_count + 1;
        end else begin 
          byte_count <= 0;
          state <= 1;
        end
      end

      1: begin /* running */
        state <= 2;
        infer_count <= infer_count + 1;
      end

      2: begin /* writing back */
        if (tx_count < TX_ITERATIONS) begin
          if (~tx_active) begin
            send <= 1;
            tx_data <= {{ZERO_PADS{1'b0}}, output_bits[tx_count*BITS_PER_VALUE +: BITS_PER_VALUE]};
            tx_count <= tx_count + 1;
          end else begin
            send <= 0;
          end
        end else begin
          tx_count <= 0;
          state <= 0;
        end
      end
    endcase
  end

  logic_network lgn (
    .x(input_bits),
    .y(output_bits)
  );

  wire tx_active;
  wire tx_done;
  uart_tx uart_tx_inst (
    .i_Clock(clk),
    .i_Tx_Byte(tx_data),
    .i_Tx_DV(send),
    .o_Tx_Serial(tx),
    .o_Tx_Done(tx_done),
    .o_Tx_Active(tx_active)
  );

endmodule
