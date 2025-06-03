# run 1000ns
Starting instruction-by-instruction (waveform-focused) verification...

=== After 'addiu $1, $0, 10' (PC was 0x00) ===
  $1 (Destination) = 10 (0x0000000a)

=== After 'addiu $2, $0, 20' (PC was 0x04) ===
  $2 (Destination) = 20 (0x00000014)

=== After 'addu $3, $1, $2' (PC was 0x08) ===
  Verifying $3 = $1 (10) + $2 (20)
    Result  $3 = 30 (0x0000001e) (Expected: 30)

=== After 'addiu $4, $0, 7' (PC was 0x0C) ===
  $4 (Destination) = 7 (0x00000007)

=== After 'addiu $5, $0, 13' (PC was 0x10) ===
  $5 (Destination) = 13 (0x0000000d)

=== After 'and $6, $4, $5' (PC was 0x14) ===
  Verifying $6 = $4 (7) & $5 (13)
    Result  $6 = 5 (0x00000005) (Expected: 5)

=== After NOP (PC was 0x18) ===
  Value of $6 (should still be 5) = 5 (0x00000005)

$stop called at time : 240 ns : File "D:/BaiduSyncdisk/3rd-year-spring/comArch/lab/lab5/6_single_cycle_cpu/tb.v" Line 127
INFO: [USF-XSim-96] XSim completed. Design snapshot 'tb_behav' loaded.
INFO: [USF-XSim-97] XSim simulation ran for 1000ns
launch_simulation: Time (s): cpu = 00:00:02 ; elapsed = 00:00:06 . Memory (MB): peak = 1217.926 ; gain = 0.000
