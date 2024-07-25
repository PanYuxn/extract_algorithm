\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  z[0] + z[1] + z[2] + z[3] + CG
Subject To
 R0: 6 z[0] + 2 CG >= 25
 R1: 2 z[1] + 2 CG >= 30
 R2: 2 z[2] >= 14
 R3: z[3] >= 8
Bounds
End
