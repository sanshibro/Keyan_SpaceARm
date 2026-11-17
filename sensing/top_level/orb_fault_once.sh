#!/usr/bin/env bash
set -e

PRESET="${1:-medium}"
NS="/infra_fault_injector"

case "${PRESET}" in
  normal|off|restore)
    rosparam set "${NS}/mode" normal
    ;;

  mild)
    rosparam set "${NS}/mask_side" right
    rosparam set "${NS}/mask_fraction" 0.30
    rosparam set "${NS}/soft_blur_ksize" 7
    rosparam set "${NS}/soft_gain" 1.25
    rosparam set "${NS}/soft_bias" 12
    rosparam set "${NS}/soft_noise_sigma" 5
    rosparam set "${NS}/mode" left_soft_degrade
    ;;

  occlusion)
    rosparam set "${NS}/mask_side" right
    rosparam set "${NS}/mask_fraction" 0.30
    rosparam set "${NS}/black_value" 0
    rosparam set "${NS}/mode" left_half_black
    ;;

  occlusion_strong)
    rosparam set "${NS}/mask_side" right
    rosparam set "${NS}/mask_fraction" 0.45
    rosparam set "${NS}/black_value" 0
    rosparam set "${NS}/mode" left_half_black
    ;;

  medium)
    rosparam set "${NS}/mask_side" right
    rosparam set "${NS}/mask_fraction" 0.40
    rosparam set "${NS}/soft_blur_ksize" 9
    rosparam set "${NS}/soft_gain" 1.35
    rosparam set "${NS}/soft_bias" 18
    rosparam set "${NS}/soft_noise_sigma" 8
    rosparam set "${NS}/mode" left_soft_degrade
    ;;

  strong)
    rosparam set "${NS}/mask_side" right
    rosparam set "${NS}/mask_fraction" 0.50
    rosparam set "${NS}/soft_blur_ksize" 13
    rosparam set "${NS}/soft_gain" 1.55
    rosparam set "${NS}/soft_bias" 28
    rosparam set "${NS}/soft_noise_sigma" 12
    rosparam set "${NS}/mode" left_soft_degrade
    ;;

  stereo_medium)
    rosparam set "${NS}/mask_side" right
    rosparam set "${NS}/mask_fraction" 0.35
    rosparam set "${NS}/soft_blur_ksize" 9
    rosparam set "${NS}/soft_gain" 1.30
    rosparam set "${NS}/soft_bias" 15
    rosparam set "${NS}/soft_noise_sigma" 7
    rosparam set "${NS}/mode" soft_degrade
    ;;

  *)
    echo "Usage: $0 {occlusion|occlusion_strong|mild|medium|strong|stereo_medium|normal}" >&2
    exit 2
    ;;
esac

echo "ORB fault preset applied: ${PRESET}"
rosparam get "${NS}/mode"
