{
  "targets": [
    {
      "target_name": "arbor_engine",
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "cflags_cc": [ 
        "-std=c++20", 
        "-O3", 
        "-march=native",
        "-ffast-math",
        "-funroll-loops"
      ],
      "sources": [
        "src/node_bindings.cpp",
        "src/orderbook.cpp",
        "src/options_pricing.cpp",
        "src/monte_carlo.cpp",
        "src/technical_indicators.cpp",
        "src/market_data_parser.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "include"
      ],
      "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ],
      "conditions": [
        ["OS=='win'", {
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1,
              "AdditionalOptions": [ "/std:c++20", "/O2", "/GL", "/arch:AVX2" ]
            },
            "VCLinkerTool": {
              "AdditionalOptions": [ "/LTCG" ]
            }
          }
        }],
        ["OS=='mac'", {
          "xcode_settings": {
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "CLANG_CXX_LANGUAGE_STANDARD": "c++20",
            "MACOSX_DEPLOYMENT_TARGET": "10.15"
          }
        }]
      ]
    }
  ]
}
