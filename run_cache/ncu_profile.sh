#!/bin/bash

set -euo pipefail

# NCU profile for tilelang.cache.test

# 默认参数
CACHE_DIR_DEFAULT="/home/chenxi/.tilelang/cache/mha_fwd_bhsd"
ARCH_DEFAULT="sm_90"
DEVICE_DEFAULT="cuda"
REBUILD_DEFAULT=0
OUT_DIR_DEFAULT="/home/chenxi/tilelang/cache/ncu_results"

usage() {
    echo "用法: $0 [-c CACHE_DIR] [-a ARCH] [-d DEVICE] [-b REBUILD] [-o OUT_DIR]"
    echo "  -c CACHE_DIR  : 缓存目录 (默认: ${CACHE_DIR_DEFAULT})"
    echo "  -a ARCH       : GPU 架构 (默认: ${ARCH_DEFAULT})"
    echo "  -d DEVICE     : 设备 (默认: ${DEVICE_DEFAULT})"
    echo "  -b REBUILD    : 是否重编译(0/1) (默认: ${REBUILD_DEFAULT})"
    echo "  -o OUT_DIR    : 输出目录 (默认: ${OUT_DIR_DEFAULT})"
}

CACHE_DIR="${CACHE_DIR_DEFAULT}"
ARCH="${ARCH_DEFAULT}"
DEVICE="${DEVICE_DEFAULT}"
REBUILD=${REBUILD_DEFAULT}
OUT_DIR="${OUT_DIR_DEFAULT}"

while getopts "c:a:d:b:o:h" opt; do
  case $opt in
    c) CACHE_DIR="$OPTARG" ;;
    a) ARCH="$OPTARG" ;;
    d) DEVICE="$OPTARG" ;;
    b) REBUILD="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

mkdir -p "${OUT_DIR}"

# STAMP=$(date +%Y%m%d_%H%M%S)
BASE_NAME="$(basename "${CACHE_DIR}")" # _${STAMP}
REP_PATH="${OUT_DIR}/${BASE_NAME}.ncu-rep"
DETAILS_TXT="${OUT_DIR}/${BASE_NAME}_details.txt"
SOURCE_TXT="${OUT_DIR}/${BASE_NAME}_source.txt"
SUMMARY_TXT="${OUT_DIR}/${BASE_NAME}_summary.txt"

echo "运行 NCU 分析:"
echo "  CACHE_DIR = ${CACHE_DIR}"
echo "  ARCH      = ${ARCH}"
echo "  DEVICE    = ${DEVICE}"
echo "  REBUILD   = ${REBUILD}"
echo "  OUT_DIR   = ${OUT_DIR}"

REBUILD_FLAG=""
if [[ "${REBUILD}" == "1" ]]; then
  REBUILD_FLAG="--rebuild"
fi

# 仅分析 main_kernel
KERNEL_FILTER="main_kernel"

PYTHON_BIN=$(which python)

set -x
"${PYTHON_BIN}" -V >/dev/null 2>&1 || { echo "python 未找到"; exit 1; }

# 这里使用模块形式，调用 run_cache.test，内部会处理 cache 格式与运行逻辑
ncu \
  --set full \
  --kernel-name "${KERNEL_FILTER}" \
  --force-overwrite \
  --import-source yes \
  -o "${REP_PATH}" \
  "${PYTHON_BIN}" -m run_cache.test --cache_dir "${CACHE_DIR}" \
    --stats ${REBUILD_FLAG} --arch "${ARCH}"
set +x

echo "生成报告..."
ncu -i "${REP_PATH}" --page details --print-summary per-kernel > "${DETAILS_TXT}" || true
ncu -i "${REP_PATH}" --page source --csv > "${SOURCE_TXT}"  || true
ncu -i "${REP_PATH}" --print-summary per-kernel | head -50 > "${SUMMARY_TXT}" || true

echo "完成:"
echo "  REP     : ${REP_PATH}"
echo "  DETAILS : ${DETAILS_TXT}"
echo "  SOURCE  : ${SOURCE_TXT}"
echo "  SUMMARY : ${SUMMARY_TXT}"


