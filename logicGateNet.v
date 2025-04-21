module logicGateNet #(
  parameter INPUT_BITS = 400,
  parameter OUTPUT_BITS = 50
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
  localparam BITS_PER_VALUE = 5;
  localparam TX_ITERATIONS = OUTPUT_BITS / BITS_PER_VALUE;

  reg [15:0] tx_count = 0;
  reg [7:0] state = 0;
  reg signed [15:0] byte_count = INPUT_BYTES - 1;

  always @(posedge clk) begin
    case(state)
      0: begin /* reading */
        if (rx_ready && byte_count >= 0) begin
          input_bits[byte_count * 8 +: 8] <= rx_data;
          byte_count <= byte_count - 1;
        end

        if (byte_count < 0) begin
          byte_count <= INPUT_BYTES - 1;
          state <= 1;
        end
      end

      1: begin /* running */
        state <= 2;
      end

      2: begin /* writing back */
        if (tx_count == TX_ITERATIONS - 1) begin
          tx_count <= 0;
          state <= 0;
        end
        if (~tx_active) begin
          send <= 1;
          tx_data <= {3'b000, output_bits[tx_count*5 +: 5]};
        end else begin
          send <= 0;
        end
        if (tx_done) begin
          tx_count <= tx_count + 1;
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
