# Edge-LLM C++ 소스 빌드 바이너리(llm_build, llm_inference 등) 실행용 환경변수.
# 사용법: source /home/user/ondeviceAI/edgellm_check/scripts/env_cpp.sh
R=/home/user/ondeviceAI
export TRT_PACKAGE_DIR=$R/toolchain/tensorrt-11.4
export LD_LIBRARY_PATH=$R/toolchain/tensorrt-11.4/lib:$R/toolchain/cuda-13.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export EDGELLM_PLUGIN_PATH=$R/edgellm/build/libNvInfer_edgellm_plugin.so
export EDGELLM_BIN=$R/edgellm/build/examples
