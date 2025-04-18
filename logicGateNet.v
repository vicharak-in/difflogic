module logicGateNet(
    input clk,
    input rx,
    output tx
);

    wire [7:0] rx_data;
    wire rx_ready;
    reg [399:0] input_bits;
    wire [49:0] output_bits;
    reg [7:0] tx_data;
    reg send;

    // Counters and flags
    reg data_pass_flag = 1'b0;
    integer byte_count     = 0;
    integer byte2_count    = 0;
    integer delay_counter  = 0;

    // UART Receiver
    uart_rx uart_rx_inst (
        .i_Clock(clk),
        .i_Rx_Serial(rx),
        .o_Rx_Byte(rx_data),
        .o_Rx_DV(rx_ready)
    );

    // Data collection and transmission control
    always @(posedge clk) begin
        // Receive 50 bytes into input_bits
        if (rx_ready && byte_count < 50) begin
            input_bits[(399 - 8*byte_count) -: 8] <= rx_data;
            byte_count = byte_count + 1;
        end

        // When reception complete, set flag and initialize delay
        if (byte_count == 50 && !data_pass_flag) begin
            data_pass_flag <= 1'b1;
            delay_counter  <= 0;
        end

        // Wait one cycle before starting to load tx_data
        if (data_pass_flag && delay_counter < 1) begin
            delay_counter <= delay_counter + 1;
        end

        // After one cycle delay, start loading output bytes
        if (data_pass_flag && delay_counter >= 1 && byte2_count < 7) begin
            tx_data <= output_bits[(49 - 8*byte2_count) -: 8];
            byte2_count = byte2_count + 1;
        end

        // Once all bytes are loaded, trigger send for one cycle
        if (data_pass_flag && byte2_count == 7) begin
            send <= 1'b1;

            // Reset for next packet
            data_pass_flag <= 1'b0;
            byte_count     <= 0;
            byte2_count    <= 0;
            delay_counter  <= 0;
        end else begin
            send <= 1'b0;
        end
    end

    // Logic Network (combinational)
    logic_network logic_net_inst (
        .x(input_bits),
        .y(output_bits)
    );

    // UART Transmitter
    uart_tx uart_tx_inst (
        .i_Clock(clk),
        .i_Tx_Byte(tx_data),
        .i_Tx_DV(send),
        .o_Tx_Serial(tx)
    );

endmodule
