// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf16>
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf16>
    %3 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf16>
    %4 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf16>
    %5 = stablehlo.multiply %0, %0 : tensor<20x20xf16>
    %6 = stablehlo.subtract %4, %5 : tensor<20x20xf16>
    %7 = stablehlo.sqrt %6 : tensor<20x20xf16>
    %8 = stablehlo.add %3, %7 : tensor<20x20xf16>
    %9 = stablehlo.atan2 %0, %8 : tensor<20x20xf16>
    %10 = stablehlo.multiply %2, %9 : tensor<20x20xf16>
    %11 = stablehlo.custom_call @check.eq(%10, %1) : (tensor<20x20xf16>, tensor<20x20xf16>) -> tensor<i1>
    return %11 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0xE7BF59395CC350B84ABA454618BF9A3F1F442DB74EC3F83535BE2AC18543FC347742B136CAC32F3DCD33B7407CC57EBB9EC58341DEB8EFBF7444EC3C1243593E4D4602C6CF42B2382F46004375C0A2C03841A0B8303D583CEABBEDC354BD09458FC096C17B2737C002AC23438CC1874168BAC629E73AC03460B521B213C2464407C02140E4BF04349CBB6F445941AF403941B044A4BCCD35B1BC82BC02C45FC4CFC21AC43735FD3E07BC61C0EDB657B8643DBDB1E23C38C3C643243C6B452044B144B3B503BE173315C6D13EF9C2B7404242C03D2BC4E6418E4001C62F360E4137C2BA3E56B23ABD353E4E40F8C20F3C0E3C0141DD3C2D3B46C4483D77C13D3A92BC6AC51A40DDC34641B6BA714441C4D5C2F1BF203D88BB87C43744B8BA2DBAF9B837C5CAC41A30D239BF4498BC86BCA343404553C0A045B1BD39402DC2C4B839C252C1D14532BA8C3D4AC1DD36673D604024BB1C40E0C47739A23DF5BBB44180C1F7C2CF45C6403D42413DEE35A24504BF5947A7BE8F38DAC014C3FBC03F3CA6C1A0C40743BAC34AC0D62D7DC5C540163F07435EBCA8B7BEC10DC218C5BC426842A82E303C63BEF641FFC00CBA86BDA83CCA425743363C703C3643A2C130401EC293BDFBC51644C440D843DDB58A3C7FB10EB81FC542BCD942D7C2C036A9409845F4BEC3C0122DDAC2FAB669406CC1C9C62ABD0DC526BEFCC50DBB0236E2443CB468BEFFC2D8C32A414EC24545183E85BF1FC0E6BFB8444EC4F7BFF540B94153C1BEC2944138C11FC646C2DB47CDB0BAC134BCBC3F2A41DB3C51C547C47FC312B977B82DC05F453A438BBC55B30E4395C4BE3C0A3929B51A3CCF38E541A8C071C0A5ACBB3866C0CFBF113C8C432B2BDF3CA3C1ACC842406A33D1B724448840D9B611428642BC3D36C2BD4128461CC0B04243AB913F6DC05EB8A4374B41FEC1A244BAC1CF4788C032C1D04713B42FC3FDBF4FC152BC83BE0147003F92414D41DF40B8C45C431B457DC00341443E80C070431FC481C5CFC0A1BA41C5EDBA69422041B94098351A426CC3C4C1404205BA1EC2774924C237BD12C29E4282C47B4108BA24B8D2C46524B43A3CB4343A7D38A4A348C4A83E89C22342A9C19AC4A944"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
  func.func private @expected() -> tensor<20x20xf16> {
    %0 = stablehlo.constant dense<"0x00FEDB3900FE8EB83CBB00FE00FE00FE00FE71B700FE1E3600FE00FE00FE123500FEE73600FE00FEE13300FE00FEDABC00FE00FE3CB900FE00FE00FE00FE00FE00FE00FE00FE053900FE00FE00FE00FE00FEEEB800FE00FEB2BD00FE00FE00FE00FE00FE7B2700FE03AC00FE00FE00FE6DBBC6292A3CD3347CB52BB200FE00FE00FE00FE00FE0F3407BD00FE00FE00FE00FE00FE00FEF03500FE00FE00FE00FE00FE00FE503500FE00FE00FE2AB796B800FEC5B100FE00FE00FE00FE00FE00FE00FED4B500FE273300FE00FE00FE00FE00FE00FE00FE00FE00FE00FE5A3600FE00FE00FE61B200FE00FE00FE00FE00FE00FE00FE00FE743C00FE00FE00FE283B00FE00FE00FE00FE00FEF6BB00FE00FE00FE00FE00FEE8BC00FE00FEFABB0EBB5EB900FE00FE1D30853A00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE1BB900FE00FE00FE16BB00FE00FE183700FE00FE69BC00FE00FE043A00FEDEBD00FE00FE00FE00FE00FE00FE00FE133600FE00FE00FE00FED93800FE00FE00FE00FE00FE00FE00FE00FE00FED82D00FE00FE00FE00FE00FEFCB700FE00FE00FE00FE00FEAB2E00FE00FE00FE00FEDBBA00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE01B600FE86B140B800FE00FE00FE00FEF83600FE00FE00FE00FE142D00FE38B700FE00FE00FE00FE00FE00FE00FE50BC293600FE49B400FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FED2B000FE00FE00FE00FE00FE00FE00FE00FE7EB9BCB800FE00FE00FE00FE67B300FE00FE00FE733941B500FE293900FE00FE00FEA6AC103900FE00FE00FE00FE2C2B00FE00FE00FE00FE7C3315B800FE00FE13B700FE00FE00FE00FE00FE00FE00FE00FE44AB00FE00FE9FB8F73700FE00FE00FE00FE00FE00FE00FE00FE1FB400FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FE00FECFBB00FE30BC00FE00FE00FEB63500FE00FE00FE00FED1BA00FE00FE00FE00FE00FE00FE00FE00FED5BA5AB800FE6524F23B49B4193BC438A4A300FE00FE00FE00FE00FE00FE00FE"> : tensor<20x20xf16>
    return %0 : tensor<20x20xf16>
  }
}

