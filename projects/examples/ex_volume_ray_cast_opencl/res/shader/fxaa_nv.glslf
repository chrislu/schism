
#version 330 core

#if 0 // FXAA3

#extension GL_ARB_gpu_shader5 : enable

#define FXAA_PC         1
#define FXAA_GLSL_130   1
#define FXAA_LINEAR     1

//-- nvidia fxaa3 source -----------------------------------------------------
/*============================================================================


                     NVIDIA FXAA III.8 by TIMOTHY LOTTES


------------------------------------------------------------------------------                       
COPYRIGHT (C) 2010, 2011 NVIDIA CORPORATION. ALL RIGHTS RESERVED.
------------------------------------------------------------------------------                       
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED 
*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS 
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF 
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL NVIDIA 
OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR 
CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR 
LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, 
OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR INABILITY TO USE 
THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH 
DAMAGES.


------------------------------------------------------------------------------
                            INTEGRATION CHECKLIST
------------------------------------------------------------------------------
(1.) 
In the shader source,
setup defines for the desired configuration. 
Example,

  #define FXAA_PC 1
  #define FXAA_HLSL_3 1
  #define FXAA_LINEAR 1

(2.) 
Then include this file,

  #include "Fxaa3.h"

(3.) 
Then call the FXAA pixel shader from within your desired shader,

  return FxaaPixelShader(pos, posPos, tex, rcpFrame, rcpFrameOpt);

(4.) 
Insure pass prior to FXAA outputs RGBL.
See next section.

(5.) 
Setup engine to provide "rcpFrame" and "rcpFrameOpt" constants.
Not using constants will result in a performance loss.

  // {x_} = 1.0/screenWidthInPixels
  // {_y} = 1.0/screenHeightInPixels
  float2 rcpFrame

  // This must be from a constant/uniform.
  // {x___} = 2.0/screenWidthInPixels
  // {_y__} = 2.0/screenHeightInPixels
  // {__z_} = 0.5/screenWidthInPixels
  // {___w} = 0.5/screenHeightInPixels
  float4 rcpFrameOpt 
    
(6.) 
Have FXAA vertex shader run as a full screen triangle,
and output "pos" and "posPos" such that inputs in the pixel shader provide,
  
  // {xy} = center of pixel
  float2 pos,
    
  // {xy__} = upper left of pixel
  // {__zw} = lower right of pixel
  float4 posPos,

(7.) 
Insure the texture sampler used by FXAA is set to bilinear filtering.


------------------------------------------------------------------------------
                     INTEGRATION - RGBL AND COLORSPACE
------------------------------------------------------------------------------
FXAA3 requires RGBL as input.

RGB should be LDR (low dynamic range). 
Specifically do FXAA after tonemapping.

RGB data as returned by a texture fetch can be linear or non-linear.
Note an "sRGB format" texture counts as linear, 
because the result of a texture fetch is linear data.
Regular "RGBA8" textures in the sRGB colorspace are non-linear.
If a texture fetch results linear data the following is required, 

  #define FXAA_LINEAR 1 

Otherwise,

  #define FXAA_LINEAR 0

Luma must be stored in the alpha channel prior to running FXAA.
This luma value must be gamma 2.0 encoded if using FXAA_LINEAR 1.
If using FXAA_LINEAR 0, luma should match the perceptual space used for RGB.

Example pass before FXAA where output is gamma 2.0 encoded,

  color.rgb = ToneMap(color.rgb); // linear color output
  color.rgb = sqrt(color.rgb);    // gamma 2.0 color output
  return color;

To use FXAA,

  color.rgb = ToneMap(color.rgb);  // linear color output
  color.rgb = sqrt(color.rgb);     // gamma 2.0 color output
  color.a = dot(color.rgb, float3(0.299, 0.587, 0.114)); // compute luma
  return color;

Another example where output is linear encoded,
say for instance writing to an sRGB formated render target,
where the render target does the conversion back to sRGB after blending,

  color.rgb = ToneMap(color.rgb); // linear color output
  return color;

To use FXAA,

  color.rgb = ToneMap(color.rgb); // linear color output
  color.a = sqrt(dot(color.rgb, float3(0.299, 0.587, 0.114))); // compute luma
  return color;

Getting luma correct is required for the algorithm to work correctly.


------------------------------------------------------------------------------
                            COMPLEX INTEGRATION
------------------------------------------------------------------------------
Q. What if the engine is blending into RGB before wanting to run FXAA?

A. In the last opaque pass prior to FXAA,
   have the pass write out luma into alpha.
   Then blend into RGB only.
   FXAA should be able to run ok 
   assuming the blending pass did not any add aliasing.
   This should be the common case for particles and common blending passes.

============================================================================*/

/*============================================================================

                            INTEGRATION KNOBS

============================================================================*/
//
// FXAA_PS3 and FXAA_360 choose the console algorithm (FXAA3 CONSOLE).
//
// 1 = Use API.
// 0 = Don't use API.
//
/*--------------------------------------------------------------------------*/
#ifndef FXAA_PS3
    #define FXAA_PS3 0
#endif
/*--------------------------------------------------------------------------*/
#ifndef FXAA_360
    #define FXAA_360 0
#endif
/*==========================================================================*/
#ifndef FXAA_PC
    //
    // FXAA Quality 
    // The high quality PC algorithm.
    //
    #define FXAA_PC 0
#endif
/*--------------------------------------------------------------------------*/
#ifndef FXAA_PC_CONSOLE
    //
    // The console algorithm for PC is included 
    // for developers targeting really low spec machines.
    //
    #define FXAA_PC_CONSOLE 0
#endif
/*--------------------------------------------------------------------------*/
#ifndef FXAA_GLSL_120
    #define FXAA_GLSL_120 0
#endif
/*--------------------------------------------------------------------------*/
#ifndef FXAA_GLSL_130
    #define FXAA_GLSL_130 0
#endif
/*--------------------------------------------------------------------------*/
#ifndef FXAA_HLSL_3
    #define FXAA_HLSL_3 0
#endif
/*--------------------------------------------------------------------------*/
#ifndef FXAA_HLSL_4
    #define FXAA_HLSL_4 0
#endif    
/*--------------------------------------------------------------------------*/
#ifndef FXAA_HLSL_5
    #define FXAA_HLSL_5 0
#endif    
/*==========================================================================*/
#ifndef FXAA_EARLY_EXIT
    // 
    // Controls algorithm's early exit path.
    // On PS3 turning this on adds 2 cycles to the shader.
    // On 360 turning this off adds 10ths of a millisecond to the shader.
    // Turning this off on console will result in a more blurry image.
    // So this defaults to on.
    // 
    // 1 = On.
    // 0 = Off.
    // 
    #define FXAA_EARLY_EXIT 1
#endif
/*--------------------------------------------------------------------------*/
#ifndef FXAA_DISCARD
    // 
    // Only valid for PC OpenGL currently.
    // 
    // 1 = Use discard on pixels which don't need AA.
    //     For APIs which enable concurrent TEX+ROP from same surface.
    // 0 = Return unchanged color on pixels which don't need AA.
    // 
    #define FXAA_DISCARD 0
#endif    
/*--------------------------------------------------------------------------*/
#ifndef FXAA_LINEAR
    //
    // 0 = Work in non-linear color space.
    //     Use this for standard 32-bit RGBA formats.
    //
    // 1 = Work in RGB=linear, A=non-linear luma.
    //     Use this for sRGB and FP16 formats. 
    //     Works with either FXAA_ALGORITHM = 1 or 0. 
    //
    #define FXAA_LINEAR 0
#endif
/*--------------------------------------------------------------------------*/
#ifndef FXAA_FAST_PIXEL_OFFSET
    //
    // Used for GLSL 120 only.
    //
    // 1 = GL API supports fast pixel offsets
    // 0 = do not use fast pixel offsets
    // 
    #ifdef GL_EXT_gpu_shader4
        #define FXAA_FAST_PIXEL_OFFSET 1
    #endif
    #ifdef GL_NV_gpu_shader5
        #define FXAA_FAST_PIXEL_OFFSET 1
    #endif
    #ifdef GL_ARB_gpu_shader5
        #define FXAA_FAST_PIXEL_OFFSET 1
    #endif
    #ifndef FXAA_FAST_PIXEL_OFFSET
        #define FXAA_FAST_PIXEL_OFFSET 0
    #endif
#endif
/*--------------------------------------------------------------------------*/
#ifndef FXAA_GATHER4_ALPHA
    //
    // 1 = API supports gather4 on alpha channel.
    // 0 = API does not support gather4 on alpha channel.
    //
    #if (FXAA_HLSL_5 == 1)
        #define FXAA_GATHER4_ALPHA 1
    #endif
    #ifdef GL_ARB_gpu_shader5
        #define FXAA_GATHER4_ALPHA 1
    #endif
    #ifdef GL_NV_gpu_shader5
        #define FXAA_GATHER4_ALPHA 1
    #endif
    #ifndef FXAA_GATHER4_ALPHA
        #define FXAA_GATHER4_ALPHA 0
    #endif
#endif

/*============================================================================
                         FXAA CONSOLE - TUNING KNOBS
============================================================================*/
#ifndef FXAA_CONSOLE__EDGE_SHARPNESS
    //
    // Consoles the sharpness of edges.
    // 
    // Due to the PS3 being ALU bound, 
    // there are only two safe values here: 4 and 8.
    // These options use the shaders ability to a free *|/ by 4|8.
    //
    // 8.0 is sharper
    // 4.0 is softer
    //
    #if 1 
        #define FXAA_CONSOLE__EDGE_SHARPNESS 8.0
    #else
        #define FXAA_CONSOLE__EDGE_SHARPNESS 4.0
    #endif
#endif
/*--------------------------------------------------------------------------*/
#ifndef FXAA_CONSOLE__EDGE_THRESHOLD
    //
    // The minimum amount of local contrast required to apply algorithm.
    // The console setting has a different mapping than the quality setting.
    //
    // This only applies when FXAA_EARLY_EXIT is 1.
    //
    // Due to the PS3 being ALU bound, 
    // there are only two safe values here: 0.25 and 0.125.
    // These options use the shaders ability to a free *|/ by 4|8.
    //
    // 0.125 leaves less aliasing, but is softer
    // 0.25 leaves more aliasing, and is sharper
    //
    #if 1
        #define FXAA_CONSOLE__EDGE_THRESHOLD 0.125
    #else
        #define FXAA_CONSOLE__EDGE_THRESHOLD 0.25
    #endif        
#endif
/*--------------------------------------------------------------------------*/
#ifndef FXAA_CONSOLE__EDGE_THRESHOLD_MIN
    //
    // Trims the algorithm from processing darks.
    // The console setting has a different mapping than the quality setting.
    //
    // This only applies when FXAA_EARLY_EXIT is 1.
    //
    // This does not apply to PS3.
    // PS3 was simplified to avoid more shader instructions.
    // 
    #define FXAA_CONSOLE__EDGE_THRESHOLD_MIN 0.05
#endif

/*============================================================================
                         FXAA QUALITY - TUNING KNOBS
============================================================================*/
#ifndef FXAA_QUALITY__EDGE_THRESHOLD
    //
    // The minimum amount of local contrast required to apply algorithm.
    //
    // 1/3 - too little
    // 1/4 - low quality
    // 1/6 - default
    // 1/8 - high quality
    // 1/16 - overkill
    //
    #define FXAA_QUALITY__EDGE_THRESHOLD (1.0/6.0)
#endif
/*--------------------------------------------------------------------------*/
#ifndef FXAA_QUALITY__EDGE_THRESHOLD_MIN
    //
    // Trims the algorithm from processing darks.
    //
    // 1/32 - visible limit
    // 1/16 - high quality
    // 1/12 - upper limit (default, the start of visible unfiltered edges)
    //
    #define FXAA_QUALITY__EDGE_THRESHOLD_MIN (1.0/12.0)
#endif
/*--------------------------------------------------------------------------*/
#ifndef FXAA_QUALITY__SUBPIX_CAP
    //
    // Insures fine detail is not completely removed.
    // This partly overrides FXAA_SUBPIX_TRIM.
    //
    // 3/4 - default amount of filtering
    // 7/8 - high amount of filtering
    // 1 - no capping of filtering
    //
    #define FXAA_QUALITY__SUBPIX_CAP (3.0/4.0)
#endif
/*--------------------------------------------------------------------------*/
#ifndef FXAA_QUALITY__SUBPIX_TRIM
    //
    // Controls removal of sub-pixel aliasing,
    //
    // 1/2 - low removal (sharper but more sub-pixel aliasing)
    // 1/3 - medium removal
    // 1/4 - default removal
    // 1/8 - high removal
    // 0 - complete removal (softer but less sub-pixel aliasing)
    //
    #define FXAA_QUALITY__SUBPIX_TRIM (1.0/4.0)
#endif


/*============================================================================

                                 API PORTING
                                 
============================================================================*/
#if FXAA_GLSL_120
    // Requires,
    //  #version 120
    // And at least,
    //  #extension GL_EXT_gpu_shader4 : enable
    //  (or set FXAA_FAST_PIXEL_OFFSET 1 to work like DX9) 
    #define half float
    #define half2 vec2
    #define half3 vec3
    #define half4 vec4
    #define int2 ivec2
    #define float2 vec2
    #define float3 vec3
    #define float4 vec4
    #define FxaaInt2 ivec2
    #define FxaaFloat2 vec2
    #define FxaaFloat3 vec3
    #define FxaaFloat4 vec4
    #define FxaaDiscard discard
    #define FxaaDot3(a, b) dot(a, b)
    #define FxaaSat(x) clamp(x, 0.0, 1.0)
    #define FxaaLerp(x,y,s) mix(x,y,s)
    #define FxaaTex sampler2D
    #define FxaaTexTop(t, p) texture2DLod(t, p, 0.0)
    #if (FXAA_FAST_PIXEL_OFFSET == 1)
        #define FxaaTexOff(t, p, o, r) texture2DLodOffset(t, p, 0.0, o)
    #else
        #define FxaaTexOff(t, p, o, r) texture2DLod(t, p + (o * r), 0.0)
    #endif
    #if (FXAA_GATHER4_ALPHA == 1)
        // use #extension GL_ARB_gpu_shader5 : enable
        #define FxaaTexAlpha4(t, p, r) textureGather(t, p, 3)
        #define FxaaTexOffAlpha4(t, p, o, r) textureGatherOffset(t, p, o, 3)
    #endif        
#endif
/*--------------------------------------------------------------------------*/
#if FXAA_GLSL_130
    // Requires "#version 130" or better
    #define half float
    #define half2 vec2
    #define half3 vec3
    #define half4 vec4
    #define int2 ivec2
    #define float2 vec2
    #define float3 vec3
    #define float4 vec4
    #define FxaaInt2 ivec2
    #define FxaaFloat2 vec2
    #define FxaaFloat3 vec3
    #define FxaaFloat4 vec4
    #define FxaaDiscard discard
    #define FxaaDot3(a, b) dot(a, b)
    #define FxaaSat(x) clamp(x, 0.0, 1.0)
    #define FxaaLerp(x,y,s) mix(x,y,s)
    #define FxaaTex sampler2D
    #define FxaaTexTop(t, p) textureLod(t, p, 0.0)
    #define FxaaTexOff(t, p, o, r) textureLodOffset(t, p, 0.0, o)
    #if (FXAA_GATHER4_ALPHA == 1)
        // use #extension GL_ARB_gpu_shader5 : enable
        #define FxaaTexAlpha4(t, p, r) textureGather(t, p, 3)
        #define FxaaTexOffAlpha4(t, p, o, r) textureGatherOffset(t, p, o, 3)
    #endif        
#endif
/*--------------------------------------------------------------------------*/
#if (FXAA_HLSL_3 == 1) || (FXAA_360 == 1)
    #define int2 float2
    #define FxaaInt2 float2
    #define FxaaFloat2 float2
    #define FxaaFloat3 float3
    #define FxaaFloat4 float4
    #define FxaaDiscard clip(-1)
    #define FxaaDot3(a, b) dot(a, b)
    #define FxaaSat(x) saturate(x)
    #define FxaaLerp(x,y,s) lerp(x,y,s)
    #define FxaaTex sampler2D
    #define FxaaTexTop(t, p) tex2Dlod(t, float4(p, 0.0, 0.0))
    #define FxaaTexOff(t, p, o, r) tex2Dlod(t, float4(p + (o * r), 0, 0))
#endif
/*--------------------------------------------------------------------------*/
#if FXAA_HLSL_4
    #define FxaaInt2 int2
    #define FxaaFloat2 float2
    #define FxaaFloat3 float3
    #define FxaaFloat4 float4
    #define FxaaDiscard clip(-1)
    #define FxaaDot3(a, b) dot(a, b)
    #define FxaaSat(x) saturate(x)
    #define FxaaLerp(x,y,s) lerp(x,y,s)
    struct FxaaTex { SamplerState smpl; Texture2D tex; };
    #define FxaaTexTop(t, p) t.tex.SampleLevel(t.smpl, p, 0.0) 
    #define FxaaTexOff(t, p, o, r) t.tex.SampleLevel(t.smpl, p, 0.0, o)
#endif
/*--------------------------------------------------------------------------*/
#if FXAA_HLSL_5
    #define FxaaInt2 int2
    #define FxaaFloat2 float2
    #define FxaaFloat3 float3
    #define FxaaFloat4 float4
    #define FxaaDiscard clip(-1)
    #define FxaaDot3(a, b) dot(a, b)
    #define FxaaSat(x) saturate(x)
    #define FxaaLerp(x,y,s) lerp(x,y,s)
    struct FxaaTex { SamplerState smpl; Texture2D tex; };
    #define FxaaTexTop(t, p) t.tex.SampleLevel(t.smpl, p, 0.0) 
    #define FxaaTexOff(t, p, o, r) t.tex.SampleLevel(t.smpl, p, 0.0, o)
    #define FxaaTexAlpha4(t, p, r) t.tex.GatherAlpha(t.smpl, p)
    #define FxaaTexOffAlpha4(t, p, o, r) t.tex.GatherAlpha(t.smpl, p, o)
#endif



/*============================================================================

                      FXAA3 CONSOLE - 360 PIXEL SHADER

------------------------------------------------------------------------------
Might be some optimizations left here,
as of this latest change didn't have a PIX dump to verify if TEX bound.
============================================================================*/
#if (FXAA_360 == 1)
/*--------------------------------------------------------------------------*/
half4 FxaaPixelShader(
    // {xy} = center of pixel
    float2 pos,
    // {xy__} = upper left of pixel
    // {__zw} = lower right of pixel
    float4 posPos,
    // {rgb_} = color in linear or perceptual color space
    // {___a} = alpha output is junk value
    FxaaTex tex,
    // This must be from a constant/uniform.
    // {xy} = rcpFrame not used on PC version of FXAA Console
    float2 rcpFrame,
    // This must be from a constant/uniform.
    // {x___} = 2.0/screenWidthInPixels
    // {_y__} = 2.0/screenHeightInPixels
    // {__z_} = 0.5/screenWidthInPixels
    // {___w} = 0.5/screenHeightInPixels
    float4 rcpFrameOpt 
) { 
/*--------------------------------------------------------------------------*/
    half4 lumaNwNeSwSe;
    lumaNwNeSwSe.x = FxaaTexTop(tex, posPos.xy).w;
    lumaNwNeSwSe.y = FxaaTexTop(tex, posPos.zy).w;
    lumaNwNeSwSe.z = FxaaTexTop(tex, posPos.xw).w;
    lumaNwNeSwSe.w = FxaaTexTop(tex, posPos.zw).w;
/*--------------------------------------------------------------------------*/
    half4 rgbyM = FxaaTexTop(tex, pos.xy);
/*--------------------------------------------------------------------------*/
    lumaNwNeSwSe.y += 1.0/384.0;
/*--------------------------------------------------------------------------*/
    half2 lumaMinTemp = min(lumaNwNeSwSe.xy, lumaNwNeSwSe.zw); 
    half2 lumaMaxTemp = max(lumaNwNeSwSe.xy, lumaNwNeSwSe.zw);
/*--------------------------------------------------------------------------*/
    half lumaMin = min(lumaMinTemp.x, lumaMinTemp.y);
    half lumaMax = max(lumaMaxTemp.x, lumaMaxTemp.y);
/*--------------------------------------------------------------------------*/
    half lumaMinM = min(lumaMin, rgbyM.w);
    half lumaMaxM = max(lumaMax, rgbyM.w);
    if((lumaMaxM - lumaMinM) < max(FXAA_CONSOLE__EDGE_THRESHOLD_MIN, lumaMax * FXAA_CONSOLE__EDGE_THRESHOLD))
        #if (FXAA_DISCARD == 1)
            FxaaDiscard;
        #else
            return rgbyM;
        #endif
/*--------------------------------------------------------------------------*/
    half2 dir;
    dir.x = dot(lumaNwNeSwSe, float4(-1.0, -1.0, 1.0, 1.0));
    dir.y = dot(lumaNwNeSwSe, float4( 1.0, -1.0, 1.0,-1.0));
/*--------------------------------------------------------------------------*/
    half2 dir1; 
    dir1 = normalize(dir.xy);
/*--------------------------------------------------------------------------*/
    half dirAbsMinTimesC = min(abs(dir1.x), abs(dir1.y)) * FXAA_CONSOLE__EDGE_SHARPNESS;
    half2 dir2;
    dir2 = clamp(dir1.xy / dirAbsMinTimesC, -2.0, 2.0);
/*--------------------------------------------------------------------------*/
    half4 rgbyN1 = FxaaTexTop(tex, pos.xy - dir1 * rcpFrameOpt.zw);
    half4 rgbyP1 = FxaaTexTop(tex, pos.xy + dir1 * rcpFrameOpt.zw);
    half4 rgbyN2 = FxaaTexTop(tex, pos.xy - dir2 * rcpFrameOpt.xy);
    half4 rgbyP2 = FxaaTexTop(tex, pos.xy + dir2 * rcpFrameOpt.xy);
/*--------------------------------------------------------------------------*/
    half4 rgbyA = rgbyN1 * 0.5 + rgbyP1 * 0.5;    
    half4 rgbyB = rgbyN2 * 0.25 + rgbyP2 * 0.25 + rgbyA * 0.5;
/*--------------------------------------------------------------------------*/
    bool twoTap = (rgbyB.w < lumaMin) || (rgbyB.w > lumaMax);
    if(twoTap) rgbyB.xyz = rgbyA.xyz;
    return rgbyB;
}
/*==========================================================================*/
#endif



/*============================================================================

          FXAA3 CONSOLE - OPTIMIZED PS3 PIXEL SHADER (NO EARLY EXIT)

==============================================================================
The code below does not exactly match the assembly.
I have a feeling that 12 cycles is possible, but was not able to get there.
Might have to increase register count to get full performance.
Note this shader does not use perspective interpolation.

Use the following cgc options,

  --fenable-bx2 --fastmath --fastprecision --nofloatbindings

------------------------------------------------------------------------------
                              NVSHADERPERF OUTPUT
------------------------------------------------------------------------------
For reference and to aid in debug, output of NVShaderPerf should match this,

Shader to schedule:
   0: texpkb h0.w(TRUE), v5.zyxx, #0
   2: addh h2.z(TRUE), h0.w, constant(0.001953, 0.000000, 0.000000, 0.000000).x
   4: texpkb h0.w(TRUE), v5.xwxx, #0
   6: addh h0.z(TRUE), -h2, h0.w
   7: texpkb h1.w(TRUE), v5, #0
   9: addh h0.x(TRUE), h0.z, -h1.w
  10: addh h3.w(TRUE), h0.z, h1
  11: texpkb h2.w(TRUE), v5.zwzz, #0
  13: addh h0.z(TRUE), h3.w, -h2.w
  14: addh h0.x(TRUE), h2.w, h0
  15: nrmh h1.xz(TRUE), h0_n
  16: minh_m8 h0.x(TRUE), |h1|, |h1.z|
  17: maxh h4.w(TRUE), h0, h1
  18: divx h2.xy(TRUE), h1_n.xzzw, h0_n
  19: movr r1.zw(TRUE), v4.xxxy
  20: madr r2.xz(TRUE), -h1, constant(cConst5.x, cConst5.y, cConst5.z, cConst5.w).zzww, r1.zzww
  22: minh h5.w(TRUE), h0, h1
  23: texpkb h0(TRUE), r2.xzxx, #0
  25: madr r0.zw(TRUE), h1.xzxz, constant(cConst5.x, cConst5.y, cConst5.z, cConst5.w), r1
  27: maxh h4.x(TRUE), h2.z, h2.w
  28: texpkb h1(TRUE), r0.zwzz, #0
  30: addh_d2 h1(TRUE), h0, h1
  31: madr r0.xy(TRUE), -h2, constant(cConst5.x, cConst5.y, cConst5.z, cConst5.w).xyxx, r1.zwzz
  33: texpkb h0(TRUE), r0, #0
  35: minh h4.z(TRUE), h2, h2.w
  36: fenct TRUE
  37: madr r1.xy(TRUE), h2, constant(cConst5.x, cConst5.y, cConst5.z, cConst5.w).xyxx, r1.zwzz
  39: texpkb h2(TRUE), r1, #0
  41: addh_d2 h0(TRUE), h0, h2
  42: maxh h2.w(TRUE), h4, h4.x
  43: minh h2.x(TRUE), h5.w, h4.z
  44: addh_d2 h0(TRUE), h0, h1
  45: slth h2.x(TRUE), h0.w, h2
  46: sgth h2.w(TRUE), h0, h2
  47: movh h0(TRUE), h0
  48: addx.c0 rc(TRUE), h2, h2.w
  49: movh h0(c0.NE.x), h1

IPU0 ------ Simplified schedule: --------
Pass |  Unit  |  uOp |  PC:  Op
-----+--------+------+-------------------------
   1 | SCT0/1 |  mov |   0:  TXLr h0.w, g[TEX1].zyxx, const.xxxx, TEX0;
     |    TEX |  txl |   0:  TXLr h0.w, g[TEX1].zyxx, const.xxxx, TEX0;
     |   SCB1 |  add |   2:  ADDh h2.z, h0.--w-, const.--x-;
     |        |      |
   2 | SCT0/1 |  mov |   4:  TXLr h0.w, g[TEX1].xwxx, const.xxxx, TEX0;
     |    TEX |  txl |   4:  TXLr h0.w, g[TEX1].xwxx, const.xxxx, TEX0;
     |   SCB1 |  add |   6:  ADDh h0.z,-h2, h0.--w-;
     |        |      |
   3 | SCT0/1 |  mov |   7:  TXLr h1.w, g[TEX1], const.xxxx, TEX0;
     |    TEX |  txl |   7:  TXLr h1.w, g[TEX1], const.xxxx, TEX0;
     |   SCB0 |  add |   9:  ADDh h0.x, h0.z---,-h1.w---;
     |   SCB1 |  add |  10:  ADDh h3.w, h0.---z, h1;
     |        |      |
   4 | SCT0/1 |  mov |  11:  TXLr h2.w, g[TEX1].zwzz, const.xxxx, TEX0;
     |    TEX |  txl |  11:  TXLr h2.w, g[TEX1].zwzz, const.xxxx, TEX0;
     |   SCB0 |  add |  14:  ADDh h0.x, h2.w---, h0;
     |   SCB1 |  add |  13:  ADDh h0.z, h3.--w-,-h2.--w-;
     |        |      |
   5 |   SCT1 |  mov |  15:  NRMh h1.xz, h0;
     |    SRB |  nrm |  15:  NRMh h1.xz, h0;
     |   SCB0 |  min |  16:  MINh*8 h0.x, |h1|, |h1.z---|;
     |   SCB1 |  max |  17:  MAXh h4.w, h0, h1;
     |        |      |
   6 |   SCT0 |  div |  18:  DIVx h2.xy, h1.xz--, h0;
     |   SCT1 |  mov |  19:  MOVr r1.zw, g[TEX0].--xy;
     |   SCB0 |  mad |  20:  MADr r2.xz,-h1, const.z-w-, r1.z-w-;
     |   SCB1 |  min |  22:  MINh h5.w, h0, h1;
     |        |      |
   7 | SCT0/1 |  mov |  23:  TXLr h0, r2.xzxx, const.xxxx, TEX0;
     |    TEX |  txl |  23:  TXLr h0, r2.xzxx, const.xxxx, TEX0;
     |   SCB0 |  max |  27:  MAXh h4.x, h2.z---, h2.w---;
     |   SCB1 |  mad |  25:  MADr r0.zw, h1.--xz, const, r1;
     |        |      |
   8 | SCT0/1 |  mov |  28:  TXLr h1, r0.zwzz, const.xxxx, TEX0;
     |    TEX |  txl |  28:  TXLr h1, r0.zwzz, const.xxxx, TEX0;
     | SCB0/1 |  add |  30:  ADDh/2 h1, h0, h1;
     |        |      |
   9 |   SCT0 |  mad |  31:  MADr r0.xy,-h2, const.xy--, r1.zw--;
     |   SCT1 |  mov |  33:  TXLr h0, r0, const.zzzz, TEX0;
     |    TEX |  txl |  33:  TXLr h0, r0, const.zzzz, TEX0;
     |   SCB1 |  min |  35:  MINh h4.z, h2, h2.--w-;
     |        |      |
  10 |   SCT0 |  mad |  37:  MADr r1.xy, h2, const.xy--, r1.zw--;
     |   SCT1 |  mov |  39:  TXLr h2, r1, const.zzzz, TEX0;
     |    TEX |  txl |  39:  TXLr h2, r1, const.zzzz, TEX0;
     | SCB0/1 |  add |  41:  ADDh/2 h0, h0, h2;
     |        |      |
  11 |   SCT0 |  min |  43:  MINh h2.x, h5.w---, h4.z---;
     |   SCT1 |  max |  42:  MAXh h2.w, h4, h4.---x;
     | SCB0/1 |  add |  44:  ADDh/2 h0, h0, h1;
     |        |      |
  12 |   SCT0 |  set |  45:  SLTh h2.x, h0.w---, h2;
     |   SCT1 |  set |  46:  SGTh h2.w, h0, h2;
     | SCB0/1 |  mul |  47:  MOVh h0, h0;
     |        |      |
  13 |   SCT0 |  mad |  48:  ADDxc0_s rc, h2, h2.w---;
     | SCB0/1 |  mul |  49:  MOVh h0(NE0.xxxx), h1;

Pass   SCT  TEX  SCB
   1:   0% 100%  25%
   2:   0% 100%  25%
   3:   0% 100%  50%
   4:   0% 100%  50%
   5:   0%   0%  50%
   6: 100%   0%  75%
   7:   0% 100%  75%
   8:   0% 100% 100%
   9:   0% 100%  25%
  10:   0% 100% 100%
  11:  50%   0% 100%
  12:  50%   0% 100%
  13:  25%   0% 100%

MEAN:  17%  61%  67%

Pass   SCT0  SCT1   TEX  SCB0  SCB1
   1:    0%    0%  100%    0%  100%
   2:    0%    0%  100%    0%  100%
   3:    0%    0%  100%  100%  100%
   4:    0%    0%  100%  100%  100%
   5:    0%    0%    0%  100%  100%
   6:  100%  100%    0%  100%  100%
   7:    0%    0%  100%  100%  100%
   8:    0%    0%  100%  100%  100%
   9:    0%    0%  100%    0%  100%
  10:    0%    0%  100%  100%  100%
  11:  100%  100%    0%  100%  100%
  12:  100%  100%    0%  100%  100%
  13:  100%    0%    0%  100%  100%

MEAN:   30%   23%   61%   76%  100%
Fragment Performance Setup: Driver RSX Compiler, GPU RSX, Flags 0x5
Results 13 cycles, 3 r regs, 923,076,923 pixels/s
============================================================================*/
#if (FXAA_PS3 == 1) && (FXAA_EARLY_EXIT == 0)
/*--------------------------------------------------------------------------*/
#pragma disablepc all
#pragma option O3
#pragma option OutColorPrec=fp16
#pragma texformat default RGBA8
/*==========================================================================*/
half4 FxaaPixelShader(
    // {xy} = center of pixel
    float2 pos,
    // {xy__} = upper left of pixel
    // {__zw} = lower right of pixel
    float4 posPos,
    // {rgb_} = color in linear or perceptual color space
    // {___a} = luma in perceptual color space (not linear) 
    sampler2D tex,
    // This must be from a constant/uniform.
    // {xy} = rcpFrame not used on PS3
    float2 rcpFrame,
    // This must be from a constant/uniform.
    // {x___} = 2.0/screenWidthInPixels
    // {_y__} = 2.0/screenHeightInPixels
    // {__z_} = 0.5/screenWidthInPixels
    // {___w} = 0.5/screenHeightInPixels
    float4 rcpFrameOpt 
) { 
/*--------------------------------------------------------------------------*/
// (1)
    half4 dir;
    half4 lumaNe = h4tex2Dlod(tex, half4(posPos.zy, 0, 0)); 
    lumaNe.w += half(1.0/512.0);
    dir.x = -lumaNe.w;
    dir.z = -lumaNe.w;
/*--------------------------------------------------------------------------*/
// (2)
    half4 lumaSw = h4tex2Dlod(tex, half4(posPos.xw, 0, 0));
    dir.x += lumaSw.w;
    dir.z += lumaSw.w;
/*--------------------------------------------------------------------------*/
// (3)
    half4 lumaNw = h4tex2Dlod(tex, half4(posPos.xy, 0, 0));
    dir.x -= lumaNw.w;
    dir.z += lumaNw.w;
/*--------------------------------------------------------------------------*/
// (4)
    half4 lumaSe = h4tex2Dlod(tex, half4(posPos.zw, 0, 0));
    dir.x += lumaSe.w;
    dir.z -= lumaSe.w;
/*--------------------------------------------------------------------------*/
// (5)
    half4 dir1_pos;
    dir1_pos.xy = normalize(dir.xyz).xz;
    half dirAbsMinTimesC = min(abs(dir1_pos.x), abs(dir1_pos.y)) * half(FXAA_CONSOLE__EDGE_SHARPNESS);
/*--------------------------------------------------------------------------*/
// (6)
    half4 dir2_pos;
    dir2_pos.xy = clamp(dir1_pos.xy / dirAbsMinTimesC, half(-2.0), half(2.0));
    dir1_pos.zw = pos.xy;
    dir2_pos.zw = pos.xy;
    half4 temp1N;
    temp1N.xy = dir1_pos.zw - dir1_pos.xy * rcpFrameOpt.zw;
/*--------------------------------------------------------------------------*/
// (7)
    temp1N = h4tex2Dlod(tex, half4(temp1N.xy, 0.0, 0.0)); 
    half4 rgby1;
    rgby1.xy = dir1_pos.zw + dir1_pos.xy * rcpFrameOpt.zw;
/*--------------------------------------------------------------------------*/
// (8)
    rgby1 = h4tex2Dlod(tex, half4(rgby1.xy, 0.0, 0.0)); 
    rgby1 = (temp1N + rgby1) * 0.5;
/*--------------------------------------------------------------------------*/
// (9)
    half4 temp2N;
    temp2N.xy = dir2_pos.zw - dir2_pos.xy * rcpFrameOpt.xy;
    temp2N = h4tex2Dlod(tex, half4(temp2N.xy, 0.0, 0.0)); 
/*--------------------------------------------------------------------------*/
// (10)
    half4 rgby2;
    rgby2.xy = dir2_pos.zw + dir2_pos.xy * rcpFrameOpt.xy;
    rgby2 = h4tex2Dlod(tex, half4(rgby2.xy, 0.0, 0.0));
    rgby2 = (temp2N + rgby2) * 0.5; 
/*--------------------------------------------------------------------------*/
// (11)
    // compilier moves these scalar ops up to other cycles
    half lumaMin = min(min(lumaNw.w, lumaSw.w), min(lumaNe.w, lumaSe.w));
    half lumaMax = max(max(lumaNw.w, lumaSw.w), max(lumaNe.w, lumaSe.w));
    rgby2 = (rgby2 + rgby1) * 0.5;
/*--------------------------------------------------------------------------*/
// (12)
    bool twoTapLt = rgby2.w < lumaMin; 
    bool twoTapGt = rgby2.w > lumaMax; 
/*--------------------------------------------------------------------------*/
// (13)
    if(twoTapLt || twoTapGt) rgby2 = rgby1;
/*--------------------------------------------------------------------------*/
    return rgby2; }
/*==========================================================================*/
#endif



/*============================================================================

        FXAA3 CONSOLE - OPTIMIZED PS3 PIXEL SHADER (WITH EARLY EXIT)

==============================================================================
The code mostly matches the assembly.
I have a feeling that 14 cycles is possible, but was not able to get there.
Might have to increase register count to get full performance.
Note this shader does not use perspective interpolation.

Use the following cgc options,

  --fenable-bx2 --fastmath --fastprecision --nofloatbindings

------------------------------------------------------------------------------
                              NVSHADERPERF OUTPUT
------------------------------------------------------------------------------
For reference and to aid in debug, output of NVShaderPerf should match this,

Shader to schedule:
   0: texpkb h0.w(TRUE), v5.zyxx, #0
   2: addh h2.y(TRUE), h0.w, constant(0.001953, 0.000000, 0.000000, 0.000000).x
   4: texpkb h1.w(TRUE), v5.xwxx, #0
   6: addh h0.x(TRUE), h1.w, -h2.y
   7: texpkb h2.w(TRUE), v5.zwzz, #0
   9: minh h4.w(TRUE), h2.y, h2
  10: maxh h5.x(TRUE), h2.y, h2.w
  11: texpkb h0.w(TRUE), v5, #0
  13: addh h3.w(TRUE), -h0, h0.x
  14: addh h0.x(TRUE), h0.w, h0
  15: addh h0.z(TRUE), -h2.w, h0.x
  16: addh h0.x(TRUE), h2.w, h3.w
  17: minh h5.y(TRUE), h0.w, h1.w
  18: nrmh h2.xz(TRUE), h0_n
  19: minh_m8 h2.w(TRUE), |h2.x|, |h2.z|
  20: divx h4.xy(TRUE), h2_n.xzzw, h2_n.w
  21: movr r1.zw(TRUE), v4.xxxy
  22: maxh h2.w(TRUE), h0, h1
  23: fenct TRUE
  24: madr r0.xy(TRUE), -h2.xzzw, constant(cConst5.x, cConst5.y, cConst5.z, cConst5.w).zwzz, r1.zwzz
  26: texpkb h0(TRUE), r0, #0
  28: maxh h5.x(TRUE), h2.w, h5
  29: minh h5.w(TRUE), h5.y, h4
  30: madr r1.xy(TRUE), h2.xzzw, constant(cConst5.x, cConst5.y, cConst5.z, cConst5.w).zwzz, r1.zwzz
  32: texpkb h2(TRUE), r1, #0
  34: addh_d2 h2(TRUE), h0, h2
  35: texpkb h1(TRUE), v4, #0
  37: maxh h5.y(TRUE), h5.x, h1.w
  38: minh h4.w(TRUE), h1, h5
  39: madr r0.xy(TRUE), -h4, constant(cConst5.x, cConst5.y, cConst5.z, cConst5.w).xyxx, r1.zwzz
  41: texpkb h0(TRUE), r0, #0
  43: addh_m8 h5.z(TRUE), h5.y, -h4.w
  44: madr r2.xy(TRUE), h4, constant(cConst5.x, cConst5.y, cConst5.z, cConst5.w).xyxx, r1.zwzz
  46: texpkb h3(TRUE), r2, #0
  48: addh_d2 h0(TRUE), h0, h3
  49: addh_d2 h3(TRUE), h0, h2
  50: movh h0(TRUE), h3
  51: slth h3.x(TRUE), h3.w, h5.w
  52: sgth h3.w(TRUE), h3, h5.x
  53: addx.c0 rc(TRUE), h3.x, h3
  54: slth.c0 rc(TRUE), h5.z, h5
  55: movh h0(c0.NE.w), h2
  56: movh h0(c0.NE.x), h1

IPU0 ------ Simplified schedule: --------
Pass |  Unit  |  uOp |  PC:  Op
-----+--------+------+-------------------------
   1 | SCT0/1 |  mov |   0:  TXLr h0.w, g[TEX1].zyxx, const.xxxx, TEX0;
     |    TEX |  txl |   0:  TXLr h0.w, g[TEX1].zyxx, const.xxxx, TEX0;
     |   SCB0 |  add |   2:  ADDh h2.y, h0.-w--, const.-x--;
     |        |      |
   2 | SCT0/1 |  mov |   4:  TXLr h1.w, g[TEX1].xwxx, const.xxxx, TEX0;
     |    TEX |  txl |   4:  TXLr h1.w, g[TEX1].xwxx, const.xxxx, TEX0;
     |   SCB0 |  add |   6:  ADDh h0.x, h1.w---,-h2.y---;
     |        |      |
   3 | SCT0/1 |  mov |   7:  TXLr h2.w, g[TEX1].zwzz, const.xxxx, TEX0;
     |    TEX |  txl |   7:  TXLr h2.w, g[TEX1].zwzz, const.xxxx, TEX0;
     |   SCB0 |  max |  10:  MAXh h5.x, h2.y---, h2.w---;
     |   SCB1 |  min |   9:  MINh h4.w, h2.---y, h2;
     |        |      |
   4 | SCT0/1 |  mov |  11:  TXLr h0.w, g[TEX1], const.xxxx, TEX0;
     |    TEX |  txl |  11:  TXLr h0.w, g[TEX1], const.xxxx, TEX0;
     |   SCB0 |  add |  14:  ADDh h0.x, h0.w---, h0;
     |   SCB1 |  add |  13:  ADDh h3.w,-h0, h0.---x;
     |        |      |
   5 |   SCT0 |  mad |  16:  ADDh h0.x, h2.w---, h3.w---;
     |   SCT1 |  mad |  15:  ADDh h0.z,-h2.--w-, h0.--x-;
     |   SCB0 |  min |  17:  MINh h5.y, h0.-w--, h1.-w--;
     |        |      |
   6 |   SCT1 |  mov |  18:  NRMh h2.xz, h0;
     |    SRB |  nrm |  18:  NRMh h2.xz, h0;
     |   SCB1 |  min |  19:  MINh*8 h2.w, |h2.---x|, |h2.---z|;
     |        |      |
   7 |   SCT0 |  div |  20:  DIVx h4.xy, h2.xz--, h2.ww--;
     |   SCT1 |  mov |  21:  MOVr r1.zw, g[TEX0].--xy;
     |   SCB1 |  max |  22:  MAXh h2.w, h0, h1;
     |        |      |
   8 |   SCT0 |  mad |  24:  MADr r0.xy,-h2.xz--, const.zw--, r1.zw--;
     |   SCT1 |  mov |  26:  TXLr h0, r0, const.xxxx, TEX0;
     |    TEX |  txl |  26:  TXLr h0, r0, const.xxxx, TEX0;
     |   SCB0 |  max |  28:  MAXh h5.x, h2.w---, h5;
     |   SCB1 |  min |  29:  MINh h5.w, h5.---y, h4;
     |        |      |
   9 |   SCT0 |  mad |  30:  MADr r1.xy, h2.xz--, const.zw--, r1.zw--;
     |   SCT1 |  mov |  32:  TXLr h2, r1, const.xxxx, TEX0;
     |    TEX |  txl |  32:  TXLr h2, r1, const.xxxx, TEX0;
     | SCB0/1 |  add |  34:  ADDh/2 h2, h0, h2;
     |        |      |
  10 | SCT0/1 |  mov |  35:  TXLr h1, g[TEX0], const.xxxx, TEX0;
     |    TEX |  txl |  35:  TXLr h1, g[TEX0], const.xxxx, TEX0;
     |   SCB0 |  max |  37:  MAXh h5.y, h5.-x--, h1.-w--;
     |   SCB1 |  min |  38:  MINh h4.w, h1, h5;
     |        |      |
  11 |   SCT0 |  mad |  39:  MADr r0.xy,-h4, const.xy--, r1.zw--;
     |   SCT1 |  mov |  41:  TXLr h0, r0, const.zzzz, TEX0;
     |    TEX |  txl |  41:  TXLr h0, r0, const.zzzz, TEX0;
     |   SCB0 |  mad |  44:  MADr r2.xy, h4, const.xy--, r1.zw--;
     |   SCB1 |  add |  43:  ADDh*8 h5.z, h5.--y-,-h4.--w-;
     |        |      |
  12 | SCT0/1 |  mov |  46:  TXLr h3, r2, const.xxxx, TEX0;
     |    TEX |  txl |  46:  TXLr h3, r2, const.xxxx, TEX0;
     | SCB0/1 |  add |  48:  ADDh/2 h0, h0, h3;
     |        |      |
  13 | SCT0/1 |  mad |  49:  ADDh/2 h3, h0, h2;
     | SCB0/1 |  mul |  50:  MOVh h0, h3;
     |        |      |
  14 |   SCT0 |  set |  51:  SLTh h3.x, h3.w---, h5.w---;
     |   SCT1 |  set |  52:  SGTh h3.w, h3, h5.---x;
     |   SCB0 |  set |  54:  SLThc0 rc, h5.z---, h5;
     |   SCB1 |  add |  53:  ADDxc0_s rc, h3.---x, h3;
     |        |      |
  15 | SCT0/1 |  mul |  55:  MOVh h0(NE0.wwww), h2;
     | SCB0/1 |  mul |  56:  MOVh h0(NE0.xxxx), h1;

Pass   SCT  TEX  SCB
   1:   0% 100%  25%
   2:   0% 100%  25%
   3:   0% 100%  50%
   4:   0% 100%  50%
   5:  50%   0%  25%
   6:   0%   0%  25%
   7: 100%   0%  25%
   8:   0% 100%  50%
   9:   0% 100% 100%
  10:   0% 100%  50%
  11:   0% 100%  75%
  12:   0% 100% 100%
  13: 100%   0% 100%
  14:  50%   0%  50%
  15: 100%   0% 100%

MEAN:  26%  60%  56%

Pass   SCT0  SCT1   TEX  SCB0  SCB1
   1:    0%    0%  100%  100%    0%
   2:    0%    0%  100%  100%    0%
   3:    0%    0%  100%  100%  100%
   4:    0%    0%  100%  100%  100%
   5:  100%  100%    0%  100%    0%
   6:    0%    0%    0%    0%  100%
   7:  100%  100%    0%    0%  100%
   8:    0%    0%  100%  100%  100%
   9:    0%    0%  100%  100%  100%
  10:    0%    0%  100%  100%  100%
  11:    0%    0%  100%  100%  100%
  12:    0%    0%  100%  100%  100%
  13:  100%  100%    0%  100%  100%
  14:  100%  100%    0%  100%  100%
  15:  100%  100%    0%  100%  100%

MEAN:   33%   33%   60%   86%   80%
Fragment Performance Setup: Driver RSX Compiler, GPU RSX, Flags 0x5
Results 15 cycles, 3 r regs, 800,000,000 pixels/s
============================================================================*/
#if (FXAA_PS3 == 1) && (FXAA_EARLY_EXIT == 1)
/*--------------------------------------------------------------------------*/
#pragma disablepc all
#pragma option O2
#pragma option OutColorPrec=fp16
#pragma texformat default RGBA8
/*==========================================================================*/
half4 FxaaPixelShader(
    // {xy} = center of pixel
    float2 pos,
    // {xy__} = upper left of pixel
    // {__zw} = lower right of pixel
    float4 posPos,
    // {rgb_} = color in linear or perceptual color space
    // {___a} = luma in perceptual color space (not linear) 
    sampler2D tex,
    // This must be from a constant/uniform.
    // {xy} = rcpFrame not used on PS3
    float2 rcpFrame,
    // This must be from a constant/uniform.
    // {x___} = 2.0/screenWidthInPixels
    // {_y__} = 2.0/screenHeightInPixels
    // {__z_} = 0.5/screenWidthInPixels
    // {___w} = 0.5/screenHeightInPixels
    float4 rcpFrameOpt 
) { 
/*--------------------------------------------------------------------------*/
// (1)
    half4 rgbyNe = h4tex2Dlod(tex, half4(posPos.zy, 0, 0)); 
    half lumaNe = rgbyNe.w + half(1.0/512.0);
/*--------------------------------------------------------------------------*/
// (2)
    half4 lumaSw = h4tex2Dlod(tex, half4(posPos.xw, 0, 0));
    half lumaSwNegNe = lumaSw.w - lumaNe;
/*--------------------------------------------------------------------------*/
// (3)
    half4 lumaNw = h4tex2Dlod(tex, half4(posPos.xy, 0, 0));
    half lumaMaxNwSw = max(lumaNw.w, lumaSw.w);
    half lumaMinNwSw = min(lumaNw.w, lumaSw.w);
/*--------------------------------------------------------------------------*/
// (4)
    half4 lumaSe = h4tex2Dlod(tex, half4(posPos.zw, 0, 0));
    half dirZ =  lumaNw.w + lumaSwNegNe; 
    half dirX = -lumaNw.w + lumaSwNegNe; 
/*--------------------------------------------------------------------------*/
// (5)
    half3 dir;
    dir.y = 0.0;
    dir.x =  lumaSe.w + dirX;
    dir.z = -lumaSe.w + dirZ;
    half lumaMinNeSe = min(lumaNe, lumaSe.w);
/*--------------------------------------------------------------------------*/
// (6)
    half4 dir1_pos;
    dir1_pos.xy = normalize(dir).xz;
    half dirAbsMinTimes8 = min(abs(dir1_pos.x), abs(dir1_pos.y)) * half(FXAA_CONSOLE__EDGE_SHARPNESS);
/*--------------------------------------------------------------------------*/
// (7)
    half4 dir2_pos;
    dir2_pos.xy = clamp(dir1_pos.xy / dirAbsMinTimes8, half(-2.0), half(2.0));
    dir1_pos.zw = pos.xy;
    dir2_pos.zw = pos.xy;
    half lumaMaxNeSe = max(lumaNe, lumaSe.w);
/*--------------------------------------------------------------------------*/
// (8)
    half4 temp1N;
    temp1N.xy = dir1_pos.zw - dir1_pos.xy * rcpFrameOpt.zw;
    temp1N = h4tex2Dlod(tex, half4(temp1N.xy, 0.0, 0.0)); 
    half lumaMax = max(lumaMaxNwSw, lumaMaxNeSe);
    half lumaMin = min(lumaMinNwSw, lumaMinNeSe);
/*--------------------------------------------------------------------------*/
// (9)
    half4 rgby1;
    rgby1.xy = dir1_pos.zw + dir1_pos.xy * rcpFrameOpt.zw;
    rgby1 = h4tex2Dlod(tex, half4(rgby1.xy, 0.0, 0.0)); 
    rgby1 = (temp1N + rgby1) * 0.5;
/*--------------------------------------------------------------------------*/
// (10)
    half4 rgbyM = h4tex2Dlod(tex, half4(pos.xy, 0.0, 0.0));
    half lumaMaxM = max(lumaMax, rgbyM.w);
    half lumaMinM = min(lumaMin, rgbyM.w);
/*--------------------------------------------------------------------------*/
// (11)
    half4 temp2N;
    temp2N.xy = dir2_pos.zw - dir2_pos.xy * rcpFrameOpt.xy;
    temp2N = h4tex2Dlod(tex, half4(temp2N.xy, 0.0, 0.0));
    half4 rgby2;
    rgby2.xy = dir2_pos.zw + dir2_pos.xy * rcpFrameOpt.xy;
    half lumaRangeM = (lumaMaxM - lumaMinM) / FXAA_CONSOLE__EDGE_THRESHOLD;
/*--------------------------------------------------------------------------*/
// (12)
    rgby2 = h4tex2Dlod(tex, half4(rgby2.xy, 0.0, 0.0));
    rgby2 = (temp2N + rgby2) * 0.5; 
/*--------------------------------------------------------------------------*/
// (13)
    rgby2 = (rgby2 + rgby1) * 0.5;
/*--------------------------------------------------------------------------*/
// (14)
    bool twoTapLt = rgby2.w < lumaMin;
    bool twoTapGt = rgby2.w > lumaMax;
    bool earlyExit = lumaRangeM < lumaMax;
    bool twoTap = twoTapLt || twoTapGt;
/*--------------------------------------------------------------------------*/
// (15)
    if(twoTap) rgby2 = rgby1;
    if(earlyExit) rgby2 = rgbyM;
/*--------------------------------------------------------------------------*/
    return rgby2; }
/*==========================================================================*/
#endif



/*============================================================================

                      FXAA3 CONSOLE - PC PIXEL SHADER

------------------------------------------------------------------------------
Using a modified version of the PS3 version here to best target old hardware.
============================================================================*/
#if (FXAA_PC_CONSOLE == 1)
/*--------------------------------------------------------------------------*/
half4 FxaaPixelShader(
    // {xy} = center of pixel
    float2 pos,
    // {xy__} = upper left of pixel
    // {__zw} = lower right of pixel
    float4 posPos,
    // {rgb_} = color in linear or perceptual color space
    // {___a} = alpha output is junk value
    FxaaTex tex,
    // This must be from a constant/uniform.
    // {xy} = rcpFrame not used on PC version of FXAA Console
    float2 rcpFrame,
    // This must be from a constant/uniform.
    // {x___} = 2.0/screenWidthInPixels
    // {_y__} = 2.0/screenHeightInPixels
    // {__z_} = 0.5/screenWidthInPixels
    // {___w} = 0.5/screenHeightInPixels
    float4 rcpFrameOpt 
) {
/*--------------------------------------------------------------------------*/
    half4 dir;
    dir.y = 0.0;
    half4 lumaNe = FxaaTexTop(tex, posPos.zy); 
    lumaNe.w += half(1.0/384.0);
    dir.x = -lumaNe.w;
    dir.z = -lumaNe.w;
/*--------------------------------------------------------------------------*/
    half4 lumaSw = FxaaTexTop(tex, posPos.xw);
    dir.x += lumaSw.w;
    dir.z += lumaSw.w;
/*--------------------------------------------------------------------------*/
    half4 lumaNw = FxaaTexTop(tex, posPos.xy);
    dir.x -= lumaNw.w;
    dir.z += lumaNw.w;
/*--------------------------------------------------------------------------*/
    half4 lumaSe = FxaaTexTop(tex, posPos.zw);
    dir.x += lumaSe.w;
    dir.z -= lumaSe.w;
/*==========================================================================*/
    #if (FXAA_EARLY_EXIT == 1)
        half4 rgbyM = FxaaTexTop(tex, pos.xy);
/*--------------------------------------------------------------------------*/
        half lumaMin = min(min(lumaNw.w, lumaSw.w), min(lumaNe.w, lumaSe.w));
        half lumaMax = max(max(lumaNw.w, lumaSw.w), max(lumaNe.w, lumaSe.w));
/*--------------------------------------------------------------------------*/
        half lumaMinM = min(lumaMin, rgbyM.w); 
        half lumaMaxM = max(lumaMax, rgbyM.w); 
/*--------------------------------------------------------------------------*/
        if((lumaMaxM - lumaMinM) < max(FXAA_CONSOLE__EDGE_THRESHOLD_MIN, lumaMax * FXAA_CONSOLE__EDGE_THRESHOLD))
            #if (FXAA_DISCARD == 1)
                FxaaDiscard;
            #else
                return rgbyM;
            #endif
    #endif
/*==========================================================================*/
    half4 dir1_pos;
    dir1_pos.xy = normalize(dir.xyz).xz;
    half dirAbsMinTimesC = min(abs(dir1_pos.x), abs(dir1_pos.y)) * half(FXAA_CONSOLE__EDGE_SHARPNESS);
/*--------------------------------------------------------------------------*/
    half4 dir2_pos;
    dir2_pos.xy = clamp(dir1_pos.xy / dirAbsMinTimesC, half(-2.0), half(2.0));
    dir1_pos.zw = pos.xy;
    dir2_pos.zw = pos.xy;
    half4 temp1N;
    temp1N.xy = dir1_pos.zw - dir1_pos.xy * rcpFrameOpt.zw;
/*--------------------------------------------------------------------------*/
    temp1N = FxaaTexTop(tex, temp1N.xy); 
    half4 rgby1;
    rgby1.xy = dir1_pos.zw + dir1_pos.xy * rcpFrameOpt.zw;
/*--------------------------------------------------------------------------*/
    rgby1 = FxaaTexTop(tex, rgby1.xy); 
    rgby1 = (temp1N + rgby1) * 0.5;
/*--------------------------------------------------------------------------*/
    half4 temp2N;
    temp2N.xy = dir2_pos.zw - dir2_pos.xy * rcpFrameOpt.xy;
    temp2N = FxaaTexTop(tex, temp2N.xy); 
/*--------------------------------------------------------------------------*/
    half4 rgby2;
    rgby2.xy = dir2_pos.zw + dir2_pos.xy * rcpFrameOpt.xy;
    rgby2 = FxaaTexTop(tex, rgby2.xy);
    rgby2 = (temp2N + rgby2) * 0.5; 
/*--------------------------------------------------------------------------*/
    #if (FXAA_EARLY_EXIT == 0)
        half lumaMin = min(min(lumaNw.w, lumaSw.w), min(lumaNe.w, lumaSe.w));
        half lumaMax = max(max(lumaNw.w, lumaSw.w), max(lumaNe.w, lumaSe.w));
    #endif
    rgby2 = (rgby2 + rgby1) * 0.5;
/*--------------------------------------------------------------------------*/
    bool twoTapLt = rgby2.w < lumaMin; 
    bool twoTapGt = rgby2.w > lumaMax; 
/*--------------------------------------------------------------------------*/
    if(twoTapLt || twoTapGt) rgby2 = rgby1;
/*--------------------------------------------------------------------------*/
    return rgby2; }
/*==========================================================================*/
#endif



/*============================================================================

                              FXAA3 QUALITY - PC

============================================================================*/
#if (FXAA_PC == 1)
/*--------------------------------------------------------------------------*/
float4 FxaaPixelShader(
    // {xy} = center of pixel
    float2 pos,
    // {xyzw} = not used on FXAA3 Quality
    float4 posPos,       
    // {rgb_} = color in linear or perceptual color space
    // {___a} = luma in perceptual color space (not linear) 
    FxaaTex tex,
    // This must be from a constant/uniform.
    // {x_} = 1.0/screenWidthInPixels
    // {_y} = 1.0/screenHeightInPixels
    float2 rcpFrame,
    // {xyzw} = not used on FXAA3 Quality
    float4 rcpFrameOpt 
) {   
/*--------------------------------------------------------------------------*/
    #if (FXAA_GATHER4_ALPHA == 1)
        float4 luma4A = FxaaTexOffAlpha4(tex, pos.xy, FxaaInt2(-1, -1), rcpFrame.xy);
        #if (FXAA_DISCARD == 0)
            float4 rgbyM = FxaaTexTop(tex, pos.xy);
        #endif
        float4 luma4B = FxaaTexAlpha4(tex, pos.xy, rcpFrame.xy);
        float lumaNE = FxaaTexOff(tex, pos.xy, FxaaInt2(1, -1), rcpFrame.xy).w;
        float lumaSW = FxaaTexOff(tex, pos.xy, FxaaInt2(-1, 1), rcpFrame.xy).w;
        float lumaNW = luma4A.w;
        float lumaN  = luma4A.z;
        float lumaW  = luma4A.x;
        float lumaM  = luma4A.y;
        float lumaE  = luma4B.z;
        float lumaS  = luma4B.x;
        float lumaSE = luma4B.y;
    #else
        float lumaN = FxaaTexOff(tex, pos.xy, FxaaInt2(0, -1), rcpFrame.xy).w;
        float lumaW = FxaaTexOff(tex, pos.xy, FxaaInt2(-1, 0), rcpFrame.xy).w;
        float4 rgbyM = FxaaTexTop(tex, pos.xy);
        float lumaE = FxaaTexOff(tex, pos.xy, FxaaInt2( 1, 0), rcpFrame.xy).w;
        float lumaS = FxaaTexOff(tex, pos.xy, FxaaInt2( 0, 1), rcpFrame.xy).w;
        float lumaM = rgbyM.w;
    #endif
/*--------------------------------------------------------------------------*/
    float rangeMin = min(lumaM, min(min(lumaN, lumaW), min(lumaS, lumaE)));
    float rangeMax = max(lumaM, max(max(lumaN, lumaW), max(lumaS, lumaE)));
    float range = rangeMax - rangeMin;
/*--------------------------------------------------------------------------*/
    if(range < max(FXAA_QUALITY__EDGE_THRESHOLD_MIN, rangeMax * FXAA_QUALITY__EDGE_THRESHOLD))
        #if (FXAA_DISCARD == 1)
            FxaaDiscard;
        #else
            return rgbyM;
        #endif
/*--------------------------------------------------------------------------*/
    #if (FXAA_GATHER4_ALPHA == 0) 
        float lumaNW = FxaaTexOff(tex, pos.xy, FxaaInt2(-1,-1), rcpFrame.xy).w;
        float lumaNE = FxaaTexOff(tex, pos.xy, FxaaInt2( 1,-1), rcpFrame.xy).w;
        float lumaSW = FxaaTexOff(tex, pos.xy, FxaaInt2(-1, 1), rcpFrame.xy).w;
        float lumaSE = FxaaTexOff(tex, pos.xy, FxaaInt2( 1, 1), rcpFrame.xy).w;
    #endif
/*--------------------------------------------------------------------------*/
    #define FXAA_QUALITY__SUBPIX_TRIM_SCALE  (1.0/(1.0 - FXAA_QUALITY__SUBPIX_TRIM))
/*--------------------------------------------------------------------------*/
    float lumaL = (lumaN + lumaW + lumaE + lumaS) * 0.25;
    float rangeL = abs(lumaL - lumaM);
    float blendL = FxaaSat((rangeL / range) - FXAA_QUALITY__SUBPIX_TRIM) * FXAA_QUALITY__SUBPIX_TRIM_SCALE; 
    blendL = min(FXAA_QUALITY__SUBPIX_CAP, blendL);
/*--------------------------------------------------------------------------*/
    float edgeVert = 
              abs(lumaNW + (-2.0 * lumaN) + lumaNE) +
        2.0 * abs(lumaW  + (-2.0 * lumaM) + lumaE ) +
              abs(lumaSW + (-2.0 * lumaS) + lumaSE);
    float edgeHorz = 
              abs(lumaNW + (-2.0 * lumaW) + lumaSW) +
        2.0 * abs(lumaN  + (-2.0 * lumaM) + lumaS ) +
              abs(lumaNE + (-2.0 * lumaE) + lumaSE);
    bool horzSpan = edgeHorz >= edgeVert;
/*--------------------------------------------------------------------------*/
    float lengthSign = horzSpan ? -rcpFrame.y : -rcpFrame.x;
    if(!horzSpan) lumaN = lumaW;
    if(!horzSpan) lumaS = lumaE;
    float gradientN = abs(lumaN - lumaM);
    float gradientS = abs(lumaS - lumaM);
    lumaN = (lumaN + lumaM) * 0.5;
    lumaS = (lumaS + lumaM) * 0.5;
/*--------------------------------------------------------------------------*/
    bool pairN = gradientN >= gradientS;
    if(!pairN) lumaN = lumaS;
    if(!pairN) gradientN = gradientS;
    if(!pairN) lengthSign *= -1.0;
    float2 posN;
    posN.x = pos.x + (horzSpan ? 0.0 : lengthSign * 0.5);
    posN.y = pos.y + (horzSpan ? lengthSign * 0.5 : 0.0);
/*--------------------------------------------------------------------------*/
    #define FXAA_SEARCH_STEPS     6
    #define FXAA_SEARCH_THRESHOLD (1.0/4.0)
/*--------------------------------------------------------------------------*/
    gradientN *= FXAA_SEARCH_THRESHOLD;
/*--------------------------------------------------------------------------*/
    float2 posP = posN;
    float2 offNP = horzSpan ? 
        FxaaFloat2(rcpFrame.x, 0.0) :
        FxaaFloat2(0.0f, rcpFrame.y); 
    float lumaEndN;
    float lumaEndP;
    bool doneN = false;
    bool doneP = false;
    posN += offNP * (-1.5);
    posP += offNP * ( 1.5);
    for(int i = 0; i < FXAA_SEARCH_STEPS; i++) {
        lumaEndN = FxaaTexTop(tex, posN.xy).w;
        lumaEndP = FxaaTexTop(tex, posP.xy).w;
        bool doneN2 = abs(lumaEndN - lumaN) >= gradientN;
        bool doneP2 = abs(lumaEndP - lumaN) >= gradientN;
        if(doneN2 && !doneN) posN += offNP;
        if(doneP2 && !doneP) posP -= offNP;
        if(doneN2 && doneP2) break;
        doneN = doneN2;
        doneP = doneP2;
        if(!doneN) posN -= offNP * 2.0;
        if(!doneP) posP += offNP * 2.0; }
/*--------------------------------------------------------------------------*/
    float dstN = horzSpan ? pos.x - posN.x : pos.y - posN.y;
    float dstP = horzSpan ? posP.x - pos.x : posP.y - pos.y;
/*--------------------------------------------------------------------------*/
    bool directionN = dstN < dstP;
    lumaEndN = directionN ? lumaEndN : lumaEndP;
/*--------------------------------------------------------------------------*/
    if(((lumaM - lumaN) < 0.0) == ((lumaEndN - lumaN) < 0.0)) 
        lengthSign = 0.0;
/*--------------------------------------------------------------------------*/
    float spanLength = (dstP + dstN);
    dstN = directionN ? dstN : dstP;
    float subPixelOffset = 0.5 + (dstN * (-1.0/spanLength));
    subPixelOffset += blendL * (1.0/8.0);
    subPixelOffset *= lengthSign;
    float3 rgbF = FxaaTexTop(tex, FxaaFloat2(
        pos.x + (horzSpan ? 0.0 : subPixelOffset),
        pos.y + (horzSpan ? subPixelOffset : 0.0))).xyz;
/*--------------------------------------------------------------------------*/
    #if (FXAA_LINEAR == 1)
        lumaL *= lumaL;
    #endif
    float lumaF = dot(rgbF, float3(0.299, 0.587, 0.114)) + (1.0/(65536.0*256.0));
    float lumaB = FxaaLerp(lumaF, lumaL, blendL);
    float scale = min(4.0, lumaB/lumaF);
    rgbF *= scale;
    return float4(rgbF, lumaM); }
/*==========================================================================*/
#endif


//----------------------------------------------------------------------------

noperspective in vec2 tex_coord;

uniform sampler2D in_texture;
uniform vec2      in_vp_size_rec;

layout(location = 0, index = 0) out vec4 out_color;

void main()
{
    //out_color.xyz = FxaaPixelShader(tex_coord, in_texture, in_vp_size_rec);
    out_color.xyz =   FxaaPixelShader(tex_coord, vec4(0.0), in_texture, in_vp_size_rec, vec4(0.0)).rgb;

    out_color.a   = 1.0;
}


#else  // FXAA2

//#extension GL_EXT_gpu_shader4 : enable


// Copyright for FXAA Source
//
// Copyright (c) 2010 NVIDIA Corporation. All rights reserved.
//
// TO  THE MAXIMUM  EXTENT PERMITTED  BY APPLICABLE  LAW, THIS SOFTWARE  IS PROVIDED
// *AS IS*  AND NVIDIA AND  ITS SUPPLIERS DISCLAIM  ALL WARRANTIES,  EITHER  EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED  TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL  NVIDIA OR ITS SUPPLIERS
// BE  LIABLE  FOR  ANY  SPECIAL,  INCIDENTAL,  INDIRECT,  OR  CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION,  DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE  USE OF OR INABILITY  TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

#define FXAA_GLSL_130 1

/*============================================================================

                                    FXAA

============================================================================*/

/*============================================================================
                                 API PORTING
============================================================================*/
#ifndef     FXAA_GLSL_120
    #define FXAA_GLSL_120 0
#endif
#ifndef     FXAA_GLSL_130
    #define FXAA_GLSL_130 0
#endif
#ifndef     FXAA_HLSL_3
    #define FXAA_HLSL_3 0
#endif
#ifndef     FXAA_HLSL_4
    #define FXAA_HLSL_4 0
#endif
/*--------------------------------------------------------------------------*/
#if FXAA_GLSL_120
    // Requires,
    //  #version 120
    //  #extension GL_EXT_gpu_shader4 : enable
    #define int2 ivec2
    #define float2 vec2
    #define float3 vec3
    #define float4 vec4
    #define FxaaBool3 bvec3
    #define FxaaInt2 ivec2
    #define FxaaFloat2 vec2
    #define FxaaFloat3 vec3
    #define FxaaFloat4 vec4
    #define FxaaBool2Float(a) mix(0.0, 1.0, (a))
    #define FxaaPow3(x, y) pow(x, y)
    #define FxaaSel3(f, t, b) mix((f), (t), (b))
    #define FxaaTex sampler2D
    #define FxaaTexLod0(t, p) texture2DLod(t, p, 0.0)
    #define FxaaTexOff(t, p, o, r) texture2DLodOffset(t, p, 0.0, o)
#endif
/*--------------------------------------------------------------------------*/
#if FXAA_GLSL_130
    // Requires "#version 130" or better
    #define int2 ivec2
    #define float2 vec2
    #define float3 vec3
    #define float4 vec4
    #define FxaaBool3 bvec3
    #define FxaaInt2 ivec2
    #define FxaaFloat2 vec2
    #define FxaaFloat3 vec3
    #define FxaaFloat4 vec4
    #define FxaaBool2Float(a) mix(0.0, 1.0, (a))
    #define FxaaPow3(x, y) pow(x, y)
    #define FxaaSel3(f, t, b) mix((f), (t), (b))
    #define FxaaTex sampler2D
    #define FxaaTexLod0(t, p) textureLod(t, p, 0.0)
    #define FxaaTexOff(t, p, o, r) textureLodOffset(t, p, 0.0, o)
#endif
/*--------------------------------------------------------------------------*/
#if FXAA_HLSL_3
    #define int2 float2
    #define FxaaInt2 float2
    #define FxaaFloat2 float2
    #define FxaaFloat3 float3
    #define FxaaFloat4 float4
    #define FxaaBool2Float(a) (a)
    #define FxaaPow3(x, y) pow(x, y)
    #define FxaaSel3(f, t, b) ((f)*(!b) + (t)*(b))
    #define FxaaTex sampler2D
    #define FxaaTexLod0(t, p) tex2Dlod(t, float4(p, 0.0, 0.0))
    #define FxaaTexOff(t, p, o, r) tex2Dlod(t, float4(p + (o * r), 0, 0))
#endif
/*--------------------------------------------------------------------------*/
#if FXAA_HLSL_4
    #define FxaaInt2 int2
    #define FxaaFloat2 float2
    #define FxaaFloat3 float3
    #define FxaaFloat4 float4
    #define FxaaBool2Float(a) (a)
    #define FxaaPow3(x, y) pow(x, y)
    #define FxaaSel3(f, t, b) ((f)*(!b) + (t)*(b))
    struct FxaaTex { SamplerState smpl; Texture2D tex; };
    #define FxaaTexLod0(t, p) t.tex.SampleLevel(t.smpl, p, 0.0)
    #define FxaaTexOff(t, p, o, r) t.tex.SampleLevel(t.smpl, p, 0.0, o)
#endif
/*--------------------------------------------------------------------------*/
#define FxaaToFloat3(a) FxaaFloat3((a), (a), (a))
float4 FxaaTexGrad(FxaaTex tex, float2 pos, float2 grad) {
    #if FXAA_GLSL_120
        return texture2DGrad(tex, pos.xy, grad, grad);
    #endif
    #if FXAA_GLSL_130
        return textureGrad(tex, pos.xy, grad, grad);
    #endif
    #if FXAA_HLSL_3
        return tex2Dgrad(tex, pos.xy, grad, grad);
    #endif
    #if FXAA_HLSL_4
        return tex.tex.SampleGrad(tex.smpl, pos.xy, grad, grad);
    #endif
}

/*============================================================================
                                 SRGB KNOBS
------------------------------------------------------------------------------
FXAA_SRGB_ROP - Set to 1 when applying FXAA to an sRGB back buffer (DX10/11).
                This will do the sRGB to linear transform,
                as ROP will expect linear color from this shader,
                and this shader works in non-linear color.
============================================================================*/
#define FXAA_SRGB_ROP 0

/*============================================================================
                                DEBUG KNOBS
------------------------------------------------------------------------------
All debug knobs draw FXAA-untouched pixels in FXAA computed luma (monochrome).

FXAA_DEBUG_PASSTHROUGH - Red for pixels which are filtered by FXAA with a
                         yellow tint on sub-pixel aliasing filtered by FXAA.
FXAA_DEBUG_HORZVERT    - Blue for horizontal edges, gold for vertical edges.
FXAA_DEBUG_PAIR        - Blue/green for the 2 pixel pair choice.
FXAA_DEBUG_NEGPOS      - Red/blue for which side of center of span.
FXAA_DEBUG_OFFSET      - Red/blue for -/+ x, gold/skyblue for -/+ y.
============================================================================*/
#ifndef     FXAA_DEBUG_PASSTHROUGH
    #define FXAA_DEBUG_PASSTHROUGH 0
#endif
#ifndef     FXAA_DEBUG_HORZVERT
    #define FXAA_DEBUG_HORZVERT    0
#endif
#ifndef     FXAA_DEBUG_PAIR
    #define FXAA_DEBUG_PAIR        0
#endif
#ifndef     FXAA_DEBUG_NEGPOS
    #define FXAA_DEBUG_NEGPOS      0
#endif
#ifndef     FXAA_DEBUG_OFFSET
    #define FXAA_DEBUG_OFFSET      0
#endif
/*--------------------------------------------------------------------------*/
#if FXAA_DEBUG_PASSTHROUGH || FXAA_DEBUG_HORZVERT || FXAA_DEBUG_PAIR
    #define FXAA_DEBUG 1
#endif
#if FXAA_DEBUG_NEGPOS || FXAA_DEBUG_OFFSET
    #define FXAA_DEBUG 1
#endif
#ifndef FXAA_DEBUG
    #define FXAA_DEBUG 0
#endif

/*============================================================================
                              COMPILE-IN KNOBS
------------------------------------------------------------------------------
FXAA_PRESET - Choose compile-in knob preset 0-5.
------------------------------------------------------------------------------
FXAA_EDGE_THRESHOLD - The minimum amount of local contrast required
                      to apply algorithm.
                      1.0/3.0  - too little
                      1.0/4.0  - good start
                      1.0/8.0  - applies to more edges
                      1.0/16.0 - overkill
------------------------------------------------------------------------------
FXAA_EDGE_THRESHOLD_MIN - Trims the algorithm from processing darks.
                          Perf optimization.
                          1.0/32.0 - visible limit (smaller isn't visible)
                          1.0/16.0 - good compromise
                          1.0/12.0 - upper limit (seeing artifacts)
------------------------------------------------------------------------------
FXAA_SEARCH_STEPS - Maximum number of search steps for end of span.
------------------------------------------------------------------------------
FXAA_SEARCH_ACCELERATION - How much to accelerate search,
                           1 - no acceleration
                           2 - skip by 2 pixels
                           3 - skip by 3 pixels
                           4 - skip by 4 pixels
------------------------------------------------------------------------------
FXAA_SEARCH_THRESHOLD - Controls when to stop searching.
                        1.0/4.0 - seems to be the best quality wise
------------------------------------------------------------------------------
FXAA_SUBPIX_FASTER - Turn on lower quality but faster subpix path.
                     Not recomended, but used in preset 0.
------------------------------------------------------------------------------
FXAA_SUBPIX - Toggle subpix filtering.
              0 - turn off
              1 - turn on
              2 - turn on full (ignores FXAA_SUBPIX_TRIM and CAP)
------------------------------------------------------------------------------
FXAA_SUBPIX_TRIM - Controls sub-pixel aliasing removal.
                   1.0/2.0 - low removal
                   1.0/3.0 - medium removal
                   1.0/4.0 - default removal
                   1.0/8.0 - high removal
                   0.0 - complete removal
------------------------------------------------------------------------------
FXAA_SUBPIX_CAP - Insures fine detail is not completely removed.
                  This is important for the transition of sub-pixel detail,
                  like fences and wires.
                  3.0/4.0 - default (medium amount of filtering)
                  7.0/8.0 - high amount of filtering
                  1.0 - no capping of sub-pixel aliasing removal
============================================================================*/
#ifndef FXAA_PRESET
    #define FXAA_PRESET 3
#endif
/*--------------------------------------------------------------------------*/
#if (FXAA_PRESET == 0)
    #define FXAA_EDGE_THRESHOLD      (1.0/4.0)
    #define FXAA_EDGE_THRESHOLD_MIN  (1.0/12.0)
    #define FXAA_SEARCH_STEPS        2
    #define FXAA_SEARCH_ACCELERATION 4
    #define FXAA_SEARCH_THRESHOLD    (1.0/4.0)
    #define FXAA_SUBPIX              1
    #define FXAA_SUBPIX_FASTER       1
    #define FXAA_SUBPIX_CAP          (2.0/3.0)
    #define FXAA_SUBPIX_TRIM         (1.0/4.0)
#endif
/*--------------------------------------------------------------------------*/
#if (FXAA_PRESET == 1)
    #define FXAA_EDGE_THRESHOLD      (1.0/8.0)
    #define FXAA_EDGE_THRESHOLD_MIN  (1.0/16.0)
    #define FXAA_SEARCH_STEPS        4
    #define FXAA_SEARCH_ACCELERATION 3
    #define FXAA_SEARCH_THRESHOLD    (1.0/4.0)
    #define FXAA_SUBPIX              1
    #define FXAA_SUBPIX_FASTER       0
    #define FXAA_SUBPIX_CAP          (3.0/4.0)
    #define FXAA_SUBPIX_TRIM         (1.0/4.0)
#endif
/*--------------------------------------------------------------------------*/
#if (FXAA_PRESET == 2)
    #define FXAA_EDGE_THRESHOLD      (1.0/8.0)
    #define FXAA_EDGE_THRESHOLD_MIN  (1.0/24.0)
    #define FXAA_SEARCH_STEPS        8
    #define FXAA_SEARCH_ACCELERATION 2
    #define FXAA_SEARCH_THRESHOLD    (1.0/4.0)
    #define FXAA_SUBPIX              1
    #define FXAA_SUBPIX_FASTER       0
    #define FXAA_SUBPIX_CAP          (3.0/4.0)
    #define FXAA_SUBPIX_TRIM         (1.0/4.0)
#endif
/*--------------------------------------------------------------------------*/
#if (FXAA_PRESET == 3)
    #define FXAA_EDGE_THRESHOLD      (1.0/8.0)
    #define FXAA_EDGE_THRESHOLD_MIN  (1.0/24.0)
    #define FXAA_SEARCH_STEPS        16
    #define FXAA_SEARCH_ACCELERATION 1
    #define FXAA_SEARCH_THRESHOLD    (1.0/4.0)
    #define FXAA_SUBPIX              1
    #define FXAA_SUBPIX_FASTER       0
    #define FXAA_SUBPIX_CAP          (3.0/4.0)
    #define FXAA_SUBPIX_TRIM         (1.0/4.0)
#endif
/*--------------------------------------------------------------------------*/
#if (FXAA_PRESET == 4)
    #define FXAA_EDGE_THRESHOLD      (1.0/8.0)
    #define FXAA_EDGE_THRESHOLD_MIN  (1.0/24.0)
    #define FXAA_SEARCH_STEPS        24
    #define FXAA_SEARCH_ACCELERATION 1
    #define FXAA_SEARCH_THRESHOLD    (1.0/4.0)
    #define FXAA_SUBPIX              1
    #define FXAA_SUBPIX_FASTER       0
    #define FXAA_SUBPIX_CAP          (3.0/4.0)
    #define FXAA_SUBPIX_TRIM         (1.0/4.0)
#endif
/*--------------------------------------------------------------------------*/
#if (FXAA_PRESET == 5)
    #define FXAA_EDGE_THRESHOLD      (1.0/8.0)
    #define FXAA_EDGE_THRESHOLD_MIN  (1.0/24.0)
    #define FXAA_SEARCH_STEPS        32
    #define FXAA_SEARCH_ACCELERATION 1
    #define FXAA_SEARCH_THRESHOLD    (1.0/4.0)
    #define FXAA_SUBPIX              1
    #define FXAA_SUBPIX_FASTER       0
    #define FXAA_SUBPIX_CAP          (3.0/4.0)
    #define FXAA_SUBPIX_TRIM         (1.0/4.0)
#endif
/*--------------------------------------------------------------------------*/
#define FXAA_SUBPIX_TRIM_SCALE (1.0/(1.0 - FXAA_SUBPIX_TRIM))

/*============================================================================
                                   HELPERS
============================================================================*/
// Return the luma, the estimation of luminance from rgb inputs.
// This approximates luma using one FMA instruction,
// skipping normalization and tossing out blue.
// FxaaLuma() will range 0.0 to 2.963210702.
float FxaaLuma(float3 rgb) {
    return rgb.y * (0.587/0.299) + rgb.x; }
/*--------------------------------------------------------------------------*/
float3 FxaaLerp3(float3 a, float3 b, float amountOfA) {
    return (FxaaToFloat3(-amountOfA) * b) +
        ((a * FxaaToFloat3(amountOfA)) + b); }
/*--------------------------------------------------------------------------*/
// Support any extra filtering before returning color.
float3 FxaaFilterReturn(float3 rgb) {
    #if FXAA_SRGB_ROP
        // Do sRGB encoded value to linear conversion.
        return FxaaSel3(
            rgb * FxaaToFloat3(1.0/12.92),
            FxaaPow3(
                rgb * FxaaToFloat3(1.0/1.055) + FxaaToFloat3(0.055/1.055),
                FxaaToFloat3(2.4)),
            rgb > FxaaToFloat3(0.04045));
    #else
        return rgb;
    #endif
}

/*============================================================================
                                VERTEX SHADER
============================================================================*/
float2 FxaaVertexShader(
// Both x and y range {-1.0 to 1.0 across screen}.
float2 inPos) {
    float2 pos;
    pos.xy = (inPos.xy * FxaaFloat2(0.5, 0.5)) + FxaaFloat2(0.5, 0.5);
    return pos; }

/*============================================================================

                                PIXEL SHADER

============================================================================*/
float3 FxaaPixelShader(
// Output of FxaaVertexShader interpolated across screen.
//  xy -> actual texture position {0.0 to 1.0}
float2 pos,
// Input texture.
FxaaTex tex,
// RCPFRAME SHOULD PIXEL SHADER CONSTANTS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// {1.0/frameWidth, 1.0/frameHeight}
float2 rcpFrame) {

/*----------------------------------------------------------------------------
            EARLY EXIT IF LOCAL CONTRAST BELOW EDGE DETECT LIMIT
------------------------------------------------------------------------------
Majority of pixels of a typical image do not require filtering,
often pixels are grouped into blocks which could benefit from early exit
right at the beginning of the algorithm.
Given the following neighborhood,

      N
    W M E
      S

If the difference in local maximum and minimum luma (contrast "range")
is lower than a threshold proportional to the maximum local luma ("rangeMax"),
then the shader early exits (no visible aliasing).
This threshold is clamped at a minimum value ("FXAA_EDGE_THRESHOLD_MIN")
to avoid processing in really dark areas.
----------------------------------------------------------------------------*/
    float3 rgbN = FxaaTexOff(tex, pos.xy, FxaaInt2( 0,-1), rcpFrame).xyz;
    float3 rgbW = FxaaTexOff(tex, pos.xy, FxaaInt2(-1, 0), rcpFrame).xyz;
    float3 rgbM = FxaaTexOff(tex, pos.xy, FxaaInt2( 0, 0), rcpFrame).xyz;
    float3 rgbE = FxaaTexOff(tex, pos.xy, FxaaInt2( 1, 0), rcpFrame).xyz;
    float3 rgbS = FxaaTexOff(tex, pos.xy, FxaaInt2( 0, 1), rcpFrame).xyz;
    float lumaN = FxaaLuma(rgbN);
    float lumaW = FxaaLuma(rgbW);
    float lumaM = FxaaLuma(rgbM);
    float lumaE = FxaaLuma(rgbE);
    float lumaS = FxaaLuma(rgbS);
    float rangeMin = min(lumaM, min(min(lumaN, lumaW), min(lumaS, lumaE)));
    float rangeMax = max(lumaM, max(max(lumaN, lumaW), max(lumaS, lumaE)));
    float range = rangeMax - rangeMin;
    #if FXAA_DEBUG
        float lumaO = lumaM / (1.0 + (0.587/0.299));
    #endif
    if(range < max(FXAA_EDGE_THRESHOLD_MIN, rangeMax * FXAA_EDGE_THRESHOLD)) {
        #if FXAA_DEBUG
            return FxaaFilterReturn(FxaaToFloat3(lumaO));
        #endif
        return FxaaFilterReturn(rgbM); }
    #if FXAA_SUBPIX > 0
        #if FXAA_SUBPIX_FASTER
            float3 rgbL = (rgbN + rgbW + rgbE + rgbS + rgbM) *
                FxaaToFloat3(1.0/5.0);
        #else
            float3 rgbL = rgbN + rgbW + rgbM + rgbE + rgbS;
        #endif
    #endif

/*----------------------------------------------------------------------------
                               COMPUTE LOWPASS
------------------------------------------------------------------------------
FXAA computes a local neighborhood lowpass value as follows,

  (N + W + E + S)/4

Then uses the ratio of the contrast range of the lowpass
and the range found in the early exit check,
as a sub-pixel aliasing detection filter.
When FXAA detects sub-pixel aliasing (such as single pixel dots),
it later blends in "blendL" amount
of a lowpass value (computed in the next section) to the final result.
----------------------------------------------------------------------------*/
    #if FXAA_SUBPIX != 0
        float lumaL = (lumaN + lumaW + lumaE + lumaS) * 0.25;
        float rangeL = abs(lumaL - lumaM);
    #endif
    #if FXAA_SUBPIX == 1
        float blendL = max(0.0,
            (rangeL / range) - FXAA_SUBPIX_TRIM) * FXAA_SUBPIX_TRIM_SCALE;
        blendL = min(FXAA_SUBPIX_CAP, blendL);
    #endif
    #if FXAA_SUBPIX == 2
        float blendL = rangeL / range;
    #endif
    #if FXAA_DEBUG_PASSTHROUGH
        #if FXAA_SUBPIX == 0
            float blendL = 0.0;
        #endif
        return FxaaFilterReturn(
            FxaaFloat3(1.0, blendL/FXAA_SUBPIX_CAP, 0.0));
    #endif

/*----------------------------------------------------------------------------
                    CHOOSE VERTICAL OR HORIZONTAL SEARCH
------------------------------------------------------------------------------
FXAA uses the following local neighborhood,

    NW N NE
    W  M  E
    SW S SE

To compute an edge amount for both vertical and horizontal directions.
Note edge detect filters like Sobel fail on single pixel lines through M.
FXAA takes the weighted average magnitude of the high-pass values
for rows and columns as an indication of local edge amount.

A lowpass value for anti-sub-pixel-aliasing is computed as
    (N+W+E+S+M+NW+NE+SW+SE)/9.
This full box pattern has higher quality than other options.

Note following this block, both vertical and horizontal cases
flow in parallel (reusing the horizontal variables).
----------------------------------------------------------------------------*/
    float3 rgbNW = FxaaTexOff(tex, pos.xy, FxaaInt2(-1,-1), rcpFrame).xyz;
    float3 rgbNE = FxaaTexOff(tex, pos.xy, FxaaInt2( 1,-1), rcpFrame).xyz;
    float3 rgbSW = FxaaTexOff(tex, pos.xy, FxaaInt2(-1, 1), rcpFrame).xyz;
    float3 rgbSE = FxaaTexOff(tex, pos.xy, FxaaInt2( 1, 1), rcpFrame).xyz;
    #if (FXAA_SUBPIX_FASTER == 0) && (FXAA_SUBPIX > 0)
        rgbL += (rgbNW + rgbNE + rgbSW + rgbSE);
        rgbL *= FxaaToFloat3(1.0/9.0);
    #endif
    float lumaNW = FxaaLuma(rgbNW);
    float lumaNE = FxaaLuma(rgbNE);
    float lumaSW = FxaaLuma(rgbSW);
    float lumaSE = FxaaLuma(rgbSE);
    float edgeVert =
        abs((0.25 * lumaNW) + (-0.5 * lumaN) + (0.25 * lumaNE)) +
        abs((0.50 * lumaW ) + (-1.0 * lumaM) + (0.50 * lumaE )) +
        abs((0.25 * lumaSW) + (-0.5 * lumaS) + (0.25 * lumaSE));
    float edgeHorz =
        abs((0.25 * lumaNW) + (-0.5 * lumaW) + (0.25 * lumaSW)) +
        abs((0.50 * lumaN ) + (-1.0 * lumaM) + (0.50 * lumaS )) +
        abs((0.25 * lumaNE) + (-0.5 * lumaE) + (0.25 * lumaSE));
    bool horzSpan = edgeHorz >= edgeVert;
    #if FXAA_DEBUG_HORZVERT
        if(horzSpan) return FxaaFilterReturn(FxaaFloat3(1.0, 0.75, 0.0));
        else         return FxaaFilterReturn(FxaaFloat3(0.0, 0.50, 1.0));
    #endif
    float lengthSign = horzSpan ? -rcpFrame.y : -rcpFrame.x;
    if(!horzSpan) lumaN = lumaW;
    if(!horzSpan) lumaS = lumaE;
    float gradientN = abs(lumaN - lumaM);
    float gradientS = abs(lumaS - lumaM);
    lumaN = (lumaN + lumaM) * 0.5;
    lumaS = (lumaS + lumaM) * 0.5;

/*----------------------------------------------------------------------------
                CHOOSE SIDE OF PIXEL WHERE GRADIENT IS HIGHEST
------------------------------------------------------------------------------
This chooses a pixel pair.
For "horzSpan == true" this will be a vertical pair,

    [N]     N
    [M] or [M]
     S     [S]

Note following this block, both {N,M} and {S,M} cases
flow in parallel (reusing the {N,M} variables).

This pair of image rows or columns is searched below
in the positive and negative direction
until edge status changes
(or the maximum number of search steps is reached).
----------------------------------------------------------------------------*/
    bool pairN = gradientN >= gradientS;
    #if FXAA_DEBUG_PAIR
        if(pairN) return FxaaFilterReturn(FxaaFloat3(0.0, 0.0, 1.0));
        else      return FxaaFilterReturn(FxaaFloat3(0.0, 1.0, 0.0));
    #endif
    if(!pairN) lumaN = lumaS;
    if(!pairN) gradientN = gradientS;
    if(!pairN) lengthSign *= -1.0;
    float2 posN;
    posN.x = pos.x + (horzSpan ? 0.0 : lengthSign * 0.5);
    posN.y = pos.y + (horzSpan ? lengthSign * 0.5 : 0.0);

/*----------------------------------------------------------------------------
                         CHOOSE SEARCH LIMITING VALUES
------------------------------------------------------------------------------
Search limit (+/- gradientN) is a function of local gradient.
----------------------------------------------------------------------------*/
    gradientN *= FXAA_SEARCH_THRESHOLD;

/*----------------------------------------------------------------------------
    SEARCH IN BOTH DIRECTIONS UNTIL FIND LUMA PAIR AVERAGE IS OUT OF RANGE
------------------------------------------------------------------------------
This loop searches either in vertical or horizontal directions,
and in both the negative and positive direction in parallel.
This loop fusion is faster than searching separately.

The search is accelerated using FXAA_SEARCH_ACCELERATION length box filter
via anisotropic filtering with specified texture gradients.
----------------------------------------------------------------------------*/
    float2 posP = posN;
    float2 offNP = horzSpan ?
        FxaaFloat2(rcpFrame.x, 0.0) :
        FxaaFloat2(0.0f, rcpFrame.y);
    float lumaEndN = lumaN;
    float lumaEndP = lumaN;
    bool doneN = false;
    bool doneP = false;
    #if FXAA_SEARCH_ACCELERATION == 1
        posN += offNP * FxaaFloat2(-1.0, -1.0);
        posP += offNP * FxaaFloat2( 1.0,  1.0);
    #endif
    #if FXAA_SEARCH_ACCELERATION == 2
        posN += offNP * FxaaFloat2(-1.5, -1.5);
        posP += offNP * FxaaFloat2( 1.5,  1.5);
        offNP *= FxaaFloat2(2.0, 2.0);
    #endif
    #if FXAA_SEARCH_ACCELERATION == 3
        posN += offNP * FxaaFloat2(-2.0, -2.0);
        posP += offNP * FxaaFloat2( 2.0,  2.0);
        offNP *= FxaaFloat2(3.0, 3.0);
    #endif
    #if FXAA_SEARCH_ACCELERATION == 4
        posN += offNP * FxaaFloat2(-2.5, -2.5);
        posP += offNP * FxaaFloat2( 2.5,  2.5);
        offNP *= FxaaFloat2(4.0, 4.0);
    #endif
    for(int i = 0; i < FXAA_SEARCH_STEPS; i++) {
        #if FXAA_SEARCH_ACCELERATION == 1
            if(!doneN) lumaEndN =
                FxaaLuma(FxaaTexLod0(tex, posN.xy).xyz);
            if(!doneP) lumaEndP =
                FxaaLuma(FxaaTexLod0(tex, posP.xy).xyz);
        #else
            if(!doneN) lumaEndN =
                FxaaLuma(FxaaTexGrad(tex, posN.xy, offNP).xyz);
            if(!doneP) lumaEndP =
                FxaaLuma(FxaaTexGrad(tex, posP.xy, offNP).xyz);
        #endif
        doneN = doneN || (abs(lumaEndN - lumaN) >= gradientN);
        doneP = doneP || (abs(lumaEndP - lumaN) >= gradientN);
        if(doneN && doneP) break;
        if(!doneN) posN -= offNP;
        if(!doneP) posP += offNP; }

/*----------------------------------------------------------------------------
               HANDLE IF CENTER IS ON POSITIVE OR NEGATIVE SIDE
------------------------------------------------------------------------------
FXAA uses the pixel's position in the span
in combination with the values (lumaEnd*) at the ends of the span,
to determine filtering.

This step computes which side of the span the pixel is on.
On negative side if dstN < dstP,

     posN        pos                      posP
      |-----------|------|------------------|
      |           |      |                  |
      |<--dstN--->|<---------dstP---------->|
                         |
                    span center

----------------------------------------------------------------------------*/
    float dstN = horzSpan ? pos.x - posN.x : pos.y - posN.y;
    float dstP = horzSpan ? posP.x - pos.x : posP.y - pos.y;
    bool directionN = dstN < dstP;
    #if FXAA_DEBUG_NEGPOS
        if(directionN) return FxaaFilterReturn(FxaaFloat3(1.0, 0.0, 0.0));
        else           return FxaaFilterReturn(FxaaFloat3(0.0, 0.0, 1.0));
    #endif
    lumaEndN = directionN ? lumaEndN : lumaEndP;

/*----------------------------------------------------------------------------
         CHECK IF PIXEL IS IN SECTION OF SPAN WHICH GETS NO FILTERING
------------------------------------------------------------------------------
If both the pair luma at the end of the span (lumaEndN)
and middle pixel luma (lumaM)
are on the same side of the middle pair average luma (lumaN),
then don't filter.

Cases,

(1.) "L",

               lumaM
                 |
                 V    XXXXXXXX <- other line averaged
         XXXXXXX[X]XXXXXXXXXXX <- source pixel line
        |      .      |
    --------------------------
       [ ]xxxxxx[x]xx[X]XXXXXX <- pair average
    --------------------------
        ^      ^ ^    ^
        |      | |    |
        .      |<---->|<---------- no filter region
        .      | |    |
        . center |    |
        .        |  lumaEndN
        .        |    .
        .      lumaN  .
        .             .
        |<--- span -->|


(2.) "^" and "-",

                               <- other line averaged
          XXXXX[X]XXX          <- source pixel line
         |     |     |
    --------------------------
        [ ]xxxx[x]xx[ ]        <- pair average
    --------------------------
         |     |     |
         |<--->|<--->|<---------- filter both sides


(3.) "v" and inverse of "-",

    XXXXXX           XXXXXXXXX <- other line averaged
    XXXXXXXXXXX[X]XXXXXXXXXXXX <- source pixel line
         |     |     |
    --------------------------
    XXXX[X]xxxx[x]xx[X]XXXXXXX <- pair average
    --------------------------
         |     |     |
         |<--->|<--->|<---------- don't filter both!


Note the "v" case for FXAA requires no filtering.
This is because the inverse of the "-" case is the "v".
Filtering "v" case turns open spans like this,

    XXXXXXXXX

Into this (which is not desired),

    x+.   .+x
    XXXXXXXXX

----------------------------------------------------------------------------*/
    if(((lumaM - lumaN) < 0.0) == ((lumaEndN - lumaN) < 0.0))
        lengthSign = 0.0;

/*----------------------------------------------------------------------------
                COMPUTE SUB-PIXEL OFFSET AND FILTER SPAN
------------------------------------------------------------------------------
FXAA filters using a bilinear texture fetch offset
from the middle pixel M towards the center of the pair (NM below).
Maximum filtering will be half way between pair.
Reminder, at this point in the code,
the {N,M} pair is also reused for all cases: {S,M}, {W,M}, and {E,M}.

    +-------+
    |       |    0.5 offset
    |   N   |     |
    |       |     V
    +-------+....---
    |       |
    |   M...|....---
    |       |     ^
    +-------+     |
    .       .    0.0 offset
    .   S   .
    .       .
    .........

Position on span is used to compute sub-pixel filter offset using simple ramp,

             posN           posP
              |\             |<------- 0.5 pixel offset into pair pixel
              | \            |
              |  \           |
    ---.......|...\..........|<------- 0.25 pixel offset into pair pixel
     ^        |   ^\         |
     |        |   | \        |
     V        |   |  \       |
    ---.......|===|==========|<------- 0.0 pixel offset (ie M pixel)
     ^        .   |   ^      .
     |        .  pos  |      .
     |        .   .   |      .
     |        .   . center   .
     |        .   .          .
     |        |<->|<---------.-------- dstN
     |        .   .          .
     |        .   |<-------->|<------- dstP
     |        .             .
     |        |<------------>|<------- spanLength
     |
    subPixelOffset

----------------------------------------------------------------------------*/
    float spanLength = (dstP + dstN);
    dstN = directionN ? dstN : dstP;
    float subPixelOffset = (0.5 + (dstN * (-1.0/spanLength))) * lengthSign;
    #if FXAA_DEBUG_OFFSET
        float ox = horzSpan ? 0.0 : subPixelOffset*2.0/rcpFrame.x;
        float oy = horzSpan ? subPixelOffset*2.0/rcpFrame.y : 0.0;
        if(ox < 0.0) return FxaaFilterReturn(
            FxaaLerp3(FxaaToFloat3(lumaO),
                      FxaaFloat3(1.0, 0.0, 0.0), -ox));
        if(ox > 0.0) return FxaaFilterReturn(
            FxaaLerp3(FxaaToFloat3(lumaO),
                      FxaaFloat3(0.0, 0.0, 1.0),  ox));
        if(oy < 0.0) return FxaaFilterReturn(
            FxaaLerp3(FxaaToFloat3(lumaO),
                      FxaaFloat3(1.0, 0.6, 0.2), -oy));
        if(oy > 0.0) return FxaaFilterReturn(
            FxaaLerp3(FxaaToFloat3(lumaO),
                      FxaaFloat3(0.2, 0.6, 1.0),  oy));
        return FxaaFilterReturn(FxaaFloat3(lumaO, lumaO, lumaO));
    #endif
    float3 rgbF = FxaaTexLod0(tex, FxaaFloat2(
        pos.x + (horzSpan ? 0.0 : subPixelOffset),
        pos.y + (horzSpan ? subPixelOffset : 0.0))).xyz;
    #if FXAA_SUBPIX == 0
        return FxaaFilterReturn(rgbF);
    #else
        return FxaaFilterReturn(FxaaLerp3(rgbL, rgbF, blendL));
    #endif
}



//----------------------------------------------------------------------------

noperspective in vec2 tex_coord;

uniform sampler2D in_texture;
uniform vec2      in_vp_size_rec;

layout(location = 0, index = 0) out vec4 out_color;

void main()
{
    out_color.xyz = FxaaPixelShader(tex_coord, in_texture, in_vp_size_rec);
    out_color.a   = 1.0;
}

#endif // FXAA 