// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<20x20xf32>
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = stablehlo.log %0 : tensor<20x20xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0xE965BBC09D5717C04591A6BF6CE2903FCA1D034009E132C0FA3A11C0C7C10EC092B67CC0B3D3543F916CCA3FA43D6D3F57424BC01904BEBFE683ACC07FBA95401815383E7E2A1B3F5F71B0C09B5C28BF39DA60BFDCFF53C0A20EEA40B399B1BCE27FE7BF0D2F273BE0F497BF7F2C46409DED943F501718C0220D63C0D7D28140874507402442B33F32111C404BFD953F661B1A40CF77CA40C480DEBE7028FDBE446F233FE41A15C08BC9D13E86F670400BFCA03F88FE1DBF53622640252548C00420BB3ECBB61FC025DC01408E8E5BBF7C3216409EF215BDEF0FD6BF9304F33F66AC9DBE7619EB3F4683A9BD3821C53DA6365440300F24405A57BFC0A7491040438C273FDECB4B404EDC3D40FD2839404859264012A9833FB22ADFBF267B5940BA664AC01BFF20BF3BC99F3F0F46F33DC94F754043C75C3F60B6F0BE524BAD3E000031C00BB71CC028CBE440E704953FEC48DC3FCD545040BF2B2B4066B291BF951113C0147788BE15173C409959833E9FC43740011E28C0F01D7E409E5801BF9B2C443F50C4463FDD5FDF3FD34CB0C0D76CF4BF728BD13FC6E67EBF9B3BA93FEAE2E6BF789B25BE915312C0C8A9DCC0CD2867407A71CA3FACBAC7407E92D73F91A8DC3F4321C6BF07BC813F902B6140821B8C40D74B96BE428567BF2746AAC0D28F383F224CD240AB90B13F1E00DB3F7C305140BFA383BFA38BEA3FB158D43EE053B2C04471FA3E64EBB63F895E8BC0CC9752BF396039BFB3027B40080A74C094F175C0BA9302C0631ADE40ABC92BBF1C23E3BEF267373F5B0CD6C04316594076993EBFFEF313C0F00B29C04A117E40D78A2140B2DB2CC0ACEDF43E46033C402B9E9DC02142C0BF9FD54C3EC099F2BF6D3BFFBF7B65C33FBB71733FBF71E7C0ED156BC072A54440431A123F6E5F823F355C224078EFF0BF306231BFA5E062BFD86F4D40959EE53F703386BFE64408C0313B8D40C6B91AC0782E0740A7D52EC0C476E03F7FC461C0D3D80C410F1B4340B54CC4BF013B9AC0F3632FC04043553FAE623D4060B9C93F9ACCEE3F5804A6409C669D406E841DC05910B9BFC1C91ABE631455C07B388940091771C078DC59407E109F3FDA1023C08D2689C0EF36B5BF47D433408CCEE0BE9BEC72BF173890C0D4AE51409D901E3EED52B740AB45AEBF80B459BFB3508740355AE93E1496C3BF936E783F1AE8DFBF7D0E8EC05DC3FFC09E1436408C4C46C04D4680BFFB636BC02B1AF9BF1F2EDD3F74C3F43F74DD903F627B3FC014F9983E88B02A3F0091CD4035351E4043DC91BFCE5233BFCEFB8ABFB0AE20C11569D93F67BD26BFF301A6BF4FBC23405CF0693EFD4F37C00E9A8F3FB4AE653F4F28D4BE2D087A3E58B312C032E8AFBEFC30354063550F4015D5BE3F4AE308BFC24888BF89B5F43F467C19C0D74B7A4031FE21BFAD1652C0DFF36C3E99385D3F448E09BE4775F0BB42406DC0C79D1F40993382C08EEBCEC0F5AA2140EBFAF9BFC0FA13C0599E2B3F055AA7BEEC0F1DC028E31D4088BB4040705827C0AF7B993FAF5CF13EDCFB90C0F6019EC0F4F6E33F6FCA6EBE3CE50340886CA83F3B428D3E349B6E3F69D8AAC02FD9CE40512DE0C07AC343C0D4B1C7C03DE9934058092340E1C8BA3EEBDDBE3FBD4A1BC0AAAC123F76F08940FD8BC440FE5909C0EA5A8ABEC5192DC09A90073FE2F0B43F2B7128404F669D401B345EBF3E6FA0404FD35A3F77376DC02B47BCC058D4CD3F68483EC0DCF2DCC079DC984067C8F33F399F183FD656D23FF84ABCBFB1D571C0AE1150C0E0A00D4074F263C0056B2AC0877D91BED3D6963ED604DBBF9C119D405BAAAABDD09FA3401B76893FFBC00C3F54E56E40005920C0B72148BFE792E9BEC1F50840EE0484BE1561A73CDA60A03F0232DD3FA7CE62C0CA70AFBF0EC07DBF13F0F5BFD3A3B6C023557AC0FC4E5B407B418640145B72BF3DDC37BECCBF343F13C5F03FD340C6BFBED70740BDAB85C06E7E38C0683EFF3FD544FEBFA13FC63D9B1F964058B17CC045A35440D586E0BFD1A6493E58EF2F40F8CEAD406C1944BF1CAC00406E0E1B3FA0C051C0B958D6BF4D925B3F66F11E40823BF1C000079E400BB75C3FAB1A97C070E90B4090E9833FCB44B13FF867114062BAD43FE6ED3BC0A2B5E03F082BAD40FB2065401296923F293977C0C0C80E40494CEC3F04D64FBF5BC36C3F637903BFB18019402CAC903D55411CC0BB2DB23FA9F0163FA9A49C3FBEF781BDFF2E86C0C2A0F6BFF69DA23F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @expected() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0xFFFFFFFFFFFFFFFFFFFFFFFF9BC2FD3D8F9A373FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7C213DBED8AAEA3E4ADB9BBDFFFFFFFFFFFFFFFFFFFFFFFFE683C53FC8A8DBBFB22C00BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBBB2FE3FFFFFFFFFFFFFFFFF9214BFC0FFFFFFFF68AC903F7C111B3EFFFFFFFFFFFFFFFFA441B33F3D973F3FC570AC3E0333643FE156223EB2F6603FEA23EC3FFFFFFFFFFFFFFFFF71C4E5BEFFFFFFFF396964BF83B2A93FE0C76A3EFFFFFFFF2096743FFFFFFFFF8CD680BFFFFFFFFF8823353FFFFFFFFF89625A3FFFFFFFFFFFFFFFFF701F243FFFFFFFFF79A41B3FFFFFFFFFF6CE15C0516F993FFDFB703FFFFFFFFFEE1B503F7E0AD9BE2641943F26308B3FB0FA873F3688743FFCFAE63CFFFFFFFFC2929C3FFFFFFFFFFFFFFFFFFC20633EFF5808C09BFCAB3F0F9217BEFFFFFFFFAEAA8ABFFFFFFFFFFFFFFFFF6BC9FB3F91B11B3EEBFA0A3F5012973FF6D87B3FFFFFFFFFFFFFFFFFFFFFFFFF29FD893F8723AEBF6703873FFFFFFFFF2C80B03FFFFFFFFFFC4788BE1B8F81BEB68B0E3FFFFFFFFFFFFFFFFF4A5EFC3EFFFFFFFF3BF98E3EFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7761A43F43B7EA3EA165EA3FAD71053FFA690B3FFFFFFFFFDE855C3C4505A13FA503BD3FFFFFFFFFFFFFFFFFFFFFFFFFDB85A7BE28FFF03FCB94A73EB07B093F0299973FFFFFFFFFDB091B3F984E61BFFFFFFFFF801037BFC8CAB63EFFFFFFFFFFFFFFFFFFFFFFFF23EDAE3FFFFFFFFFFFFFFFFFFFFFFFFFE9FCF73FFFFFFFFFFFFFFFFF3BBDAABEFFFFFFFF53579C3FFFFFFFFFFFFFFFFFFFFFFFFFCD79B03FBF066D3FFFFFFFFF8AC33CBFAEEF893FFFFFFFFFFFFFFFFF8CFCCDBFFFFFFFFFFFFFFFFF7793D83E4DFC4DBDFFFFFFFFFFFFFFFFD9AE8F3F13950FBF9177963CB3516E3FFFFFFFFFFFFFFFFFFFFFFFFFDF47953FCA9A153FFFFFFFFFFFFFFFFF6C09BE3FFFFFFFFF966B3F3FFFFFFFFF92CA0F3FFFFFFFFF99340B4020AD8E3FFFFFFFFFFFFFFFFFFFFFFFFF51093BBE0CDE8A3FD2E4E83EC8A31F3FBBBBD23F61E9CB3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF6E59BA3FFFFFFFFFFCCB9C3F647E5E3EFFFFFFFFFFFFFFFFFFFFFFFF6C3D843FFFFFFFFFFFFFFFFFFFFFFFFF3AE6973F68C2EEBF296DDF3FFFFFFFFFFFFFFFFF3C8FB83F8D2849BFFFFFFFFF82D4F5BCFFFFFFFFFFFFFFFFFFFFFFFF1CD5853FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBD040C3F82F4253F5E7CFD3DFFFFFFFFD3A19ABF8787CFBE9215EE3FD2AF673FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF269E073FFFFFFFFFFFFFFFFF897A703F10FBBCBFFFFFFFFF558CEB3DAA2ADEBDFFFFFFFF0D77B4BFFFFFFFFFFFFFFFFFB134853F1B694E3F9479CC3EFFFFFFFFFFFFFFFFF3E5253FFFFFFFFFC28FAE3FFFFFFFFFFFFFFFFFA057BBBFEB8415BEFFFFFFFFFFFFFFFFFFFFFFFFAFF4693FFFFFFFFFFFFFFFFFA0396D3FFFFFFFFFFFFFFFFF1CC0CCBEFFFFFFFFFFFFFFFFEA2A673F1C1C8D3FFFFFFFFFFBEA393ED08440BFFFFFFFFFFFFFFFFFC8C0133FFFFFFFFFD11E393F3E858C3E63D4A4BF501A90BDFFFFFFFF49E1EE3FFFFFFFFFFFFFFFFFFFFFFFFFBAF3C33F20626F3F351281BF4891CC3EFFFFFFFF0C950EBF9904BB3F5257E83FFFFFFFFFFFFFFFFFFFFFFFFF07BF22BF5339B13EC3BB773F22E9CB3FFFFFFFFFEF5ACE3F3AAB20BEFFFFFFFFFFFFFFFF8635F33EFFFFFFFFFFFFFFFF6C2AC83F67ED243F196804BF504EFE3EFFFFFFFFFFFFFFFFFFFFFFFFCD584B3FFFFFFFFFFFFFFFFFFFFFFFFF1E6E9CBFFFFFFFFF2FA4CB3FFFFFFFFF08E0D03F4C0C923D572119BF3098A83FFFFFFFFFFFFFFFFFFFFFFFFF29C4423FFFFFFFFF6E0079C0DDEA663E3D090C3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF4A49D3FAB8DB73FFFFFFFFFFFFFFFFFD535B2BE60BE213FFFFFFFFF5EAB403FFFFFFFFFFFFFFFFF37B0303FFFFFFFFF3C7215C039DAC53FFFFFFFFFC5B0993FFFFFFFFFB7FDCFBF0670813F9A9AD83FFFFFFFFF69C9323F035B00BFFFFFFFFFFFFFFFFFFF2E1DBEA2DF683FFFFFFFFF8D6BCC3F4CDD17BEFFFFFFFFA539483FE6A3F63CD4B9A63EF615523F440B023FFFFFFFFF3C12103FA621D83F5A40A33F65D60A3EFFFFFFFF716D4D3FB9F11C3FFFFFFFFFFDFB9FBDFFFFFFFF31F55F3FFE9B29C0FFFFFFFFE558A93E4F3E07BF75C94E3EFFFFFFFFFFFFFFFFFFFFFFFFCE1C753E"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
}
