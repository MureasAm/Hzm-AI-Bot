import {Config} from "@remotion/cli/config";

Config.setOverwriteOutput(true);
Config.setVideoImageFormat("jpeg");

/**
 * EP01 渲染踩到的坑：**x264 编码阶段 OOM**。
 *
 * 症状：渲染跑到 100%（18250 帧都出来了），最后 ffmpeg 拼接时报
 *   `x264 [error]: malloc of size 7088704 failed`
 *   `Conversion failed!`
 * 而且不产出文件。0.5 倍预览（960×540）不会触发——它编码器压力小得多。
 *
 * 原因：本机 15.7GB 物理内存、空闲常只剩 ~3GB（TTS 服务 + QQ + 编辑器占掉了），
 * x264 的 lookahead / rc 缓冲在 1920×1080×18250 帧下申请不到内存。
 *
 * 处置：把编码档位调轻、不用多线程切片，牺牲一点压缩率换渲染成功率。
 * 视觉质量靠 CRF 保证（不是靠 preset），所以 veryfast 对本片的画质影响很小。
 */
Config.setCrf(18);
Config.setX264Preset("veryfast");
Config.setCodec("h264");
