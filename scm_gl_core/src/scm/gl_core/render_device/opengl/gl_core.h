
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_OPENGL_DETAIL_GL_CORE_H_INCLUDED
#define SCM_GL_CORE_OPENGL_DETAIL_GL_CORE_H_INCLUDED

#include <set>
#include <string>

#include <scm/gl_core/render_device/opengl/GL/glcorearb.h>
#include <scm/gl_core/render_device/opengl/GL/glext.h>

// temporary
//#include <scm/core/platform/platform.h>
//#include <scm/core/utilities/platform_warning_disable.h>

#ifndef BUFFER_OFFSET
#define BUFFER_OFFSET(i) ((char *) NULL + (i))
#endif

namespace scm {
namespace gl {
namespace opengl {

class gl_core
{
public:
    struct context_info {
        enum profile_type {
            profile_unknown,
            profile_core = 0x01,
            profile_compatibility
        };

        int             _version_major;
        int             _version_minor;
        int             _version_release;
        std::string     _version_info;
        std::string     _glsl_version_info;

        std::string     _vendor;
        std::string     _renderer;

        profile_type    _profile;
        std::string     _profile_string;

        context_info() : _version_major(0), _version_minor(0), _version_release(0), _profile(profile_unknown) {}
    };

private:
    typedef std::set<std::string>   string_set;

public:
    gl_core();

    bool                    initialize();
    const context_info&     context_information() const;
    bool                    is_initialized() const;
    bool                    is_supported(const std::string& ext) const;
    bool                    version_supported(unsigned in_major, unsigned in_minor) const;

private:
    void            init_entry_points();

private:
    string_set      _extensions;
    bool            _initialized;
    context_info    _context_info;

    friend std::ostream& operator<<(std::ostream& out_stream, const gl_core& c);

public:
    bool version_1_0_available;
    bool version_1_1_available;
    bool version_1_2_available;
    bool version_1_3_available;
    bool version_1_4_available;
    bool version_1_5_available;
    bool version_2_0_available;
    bool version_2_1_available;
    bool version_3_0_available;
    bool version_3_1_available;
    bool version_3_2_available;
    bool version_3_3_available;
    bool version_4_0_available;
    bool version_4_1_available;
    bool version_4_2_available;
    bool version_4_3_available;
    bool version_4_4_available;

    bool extension_ARB_bindless_texture;
    bool extension_ARB_cl_event;
    bool extension_ARB_compute_variable_group_size;
    bool extension_ARB_debug_output;
    bool extension_ARB_map_buffer_alignment;
    bool extension_ARB_robustness;
    bool extension_ARB_shading_language_include;
    bool extension_ARB_sparse_texture;
    bool extension_ARB_texture_compression_bptc;

    bool extension_EXT_direct_state_access_available;
    bool extension_EXT_shader_image_load_store;
    bool extension_EXT_texture_compression_s3tc;

    bool extension_NV_bindless_texture;
    bool extension_NV_shader_buffer_load;
    bool extension_NVX_gpu_memory_info;

    bool extension_EXT_raster_multisample;
    bool extension_NV_framebuffer_mixed_samples;
    bool extension_NV_fragment_coverage_to_color;
    bool extension_NV_sample_locations;
    bool extension_NV_conservative_raster;
    bool extension_EXT_post_depth_coverage;
    bool extension_EXT_sparse_texture2;
    bool extension_NV_shader_atomic_int64;
    bool extension_NV_fragment_shader_interlock;
    bool extension_NV_sample_mask_override_coverage;
    bool extension_NV_fill_rectangle;

    // version 1.0 ////////////////////////////////////////////////////////////////////////////////
    PFNGLCULLFACEPROC                               glCullFace;
    PFNGLFRONTFACEPROC                              glFrontFace;
    PFNGLHINTPROC                                   glHint;
    PFNGLLINEWIDTHPROC                              glLineWidth;
    PFNGLPOINTSIZEPROC                              glPointSize;
    PFNGLPOLYGONMODEPROC                            glPolygonMode;
    PFNGLSCISSORPROC                                glScissor;
    PFNGLTEXPARAMETERFPROC                          glTexParameterf;
    PFNGLTEXPARAMETERFVPROC                         glTexParameterfv;
    PFNGLTEXPARAMETERIPROC                          glTexParameteri;
    PFNGLTEXPARAMETERIVPROC                         glTexParameteriv;
    PFNGLTEXIMAGE1DPROC                             glTexImage1D;
    PFNGLTEXIMAGE2DPROC                             glTexImage2D;
    PFNGLDRAWBUFFERPROC                             glDrawBuffer;
    PFNGLCLEARPROC                                  glClear;
    PFNGLCLEARCOLORPROC                             glClearColor;
    PFNGLCLEARSTENCILPROC                           glClearStencil;
    PFNGLCLEARDEPTHPROC                             glClearDepth;
    PFNGLSTENCILMASKPROC                            glStencilMask;
    PFNGLCOLORMASKPROC                              glColorMask;
    PFNGLDEPTHMASKPROC                              glDepthMask;
    PFNGLDISABLEPROC                                glDisable;
    PFNGLENABLEPROC                                 glEnable;
    PFNGLFINISHPROC                                 glFinish;
    PFNGLFLUSHPROC                                  glFlush;
    PFNGLBLENDFUNCPROC                              glBlendFunc;
    PFNGLLOGICOPPROC                                glLogicOp;
    PFNGLSTENCILFUNCPROC                            glStencilFunc;
    PFNGLSTENCILOPPROC                              glStencilOp;
    PFNGLDEPTHFUNCPROC                              glDepthFunc;
    PFNGLPIXELSTOREFPROC                            glPixelStoref;
    PFNGLPIXELSTOREIPROC                            glPixelStorei;
    PFNGLREADBUFFERPROC                             glReadBuffer;
    PFNGLREADPIXELSPROC                             glReadPixels;
    PFNGLGETBOOLEANVPROC                            glGetBooleanv;
    PFNGLGETDOUBLEVPROC                             glGetDoublev;
    PFNGLGETERRORPROC                               glGetError;
    PFNGLGETFLOATVPROC                              glGetFloatv;
    PFNGLGETINTEGERVPROC                            glGetIntegerv;
    PFNGLGETSTRINGPROC                              glGetString;
    PFNGLGETTEXIMAGEPROC                            glGetTexImage;
    PFNGLGETTEXPARAMETERFVPROC                      glGetTexParameterfv;
    PFNGLGETTEXPARAMETERIVPROC                      glGetTexParameteriv;
    PFNGLGETTEXLEVELPARAMETERFVPROC                 glGetTexLevelParameterfv;
    PFNGLGETTEXLEVELPARAMETERIVPROC                 glGetTexLevelParameteriv;
    PFNGLISENABLEDPROC                              glIsEnabled;
    PFNGLDEPTHRANGEPROC                             glDepthRange;
    PFNGLVIEWPORTPROC                               glViewport;
                                            
    // version 1.1 ////////////////////////////////////////////////////////////////////////////////
    PFNGLDRAWARRAYSPROC                             glDrawArrays;
    PFNGLDRAWELEMENTSPROC                           glDrawElements;
    PFNGLGETPOINTERVPROC                            glGetPointerv;
    PFNGLPOLYGONOFFSETPROC                          glPolygonOffset;
    PFNGLCOPYTEXIMAGE1DPROC                         glCopyTexImage1D;
    PFNGLCOPYTEXIMAGE2DPROC                         glCopyTexImage2D;
    PFNGLCOPYTEXSUBIMAGE1DPROC                      glCopyTexSubImage1D;
    PFNGLCOPYTEXSUBIMAGE2DPROC                      glCopyTexSubImage2D;
    PFNGLTEXSUBIMAGE1DPROC                          glTexSubImage1D;
    PFNGLTEXSUBIMAGE2DPROC                          glTexSubImage2D;
    PFNGLBINDTEXTUREPROC                            glBindTexture;
    PFNGLDELETETEXTURESPROC                         glDeleteTextures;
    PFNGLGENTEXTURESPROC                            glGenTextures;
    PFNGLISTEXTUREPROC                              glIsTexture;
                                            
    // version 1.2 ////////////////////////////////////////////////////////////////////////////////
    PFNGLBLENDCOLORPROC                             glBlendColor;
    PFNGLBLENDEQUATIONPROC                          glBlendEquation;
    PFNGLDRAWRANGEELEMENTSPROC                      glDrawRangeElements;
    PFNGLTEXIMAGE3DPROC                             glTexImage3D;
    PFNGLTEXSUBIMAGE3DPROC                          glTexSubImage3D;
    PFNGLCOPYTEXSUBIMAGE3DPROC                      glCopyTexSubImage3D;
                                            
    // version 1.3 ////////////////////////////////////////////////////////////////////////////////
    PFNGLACTIVETEXTUREPROC                          glActiveTexture;
    PFNGLSAMPLECOVERAGEPROC                         glSampleCoverage;
    PFNGLCOMPRESSEDTEXIMAGE3DPROC                   glCompressedTexImage3D;
    PFNGLCOMPRESSEDTEXIMAGE2DPROC                   glCompressedTexImage2D;
    PFNGLCOMPRESSEDTEXIMAGE1DPROC                   glCompressedTexImage1D;
    PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC                glCompressedTexSubImage3D;
    PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC                glCompressedTexSubImage2D;
    PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC                glCompressedTexSubImage1D;
    PFNGLGETCOMPRESSEDTEXIMAGEPROC                  glGetCompressedTexImage;
                                            
    // version 1.4 ////////////////////////////////////////////////////////////////////////////////
    PFNGLBLENDFUNCSEPARATEPROC                      glBlendFuncSeparate;
    PFNGLMULTIDRAWARRAYSPROC                        glMultiDrawArrays;
    PFNGLMULTIDRAWELEMENTSPROC                      glMultiDrawElements;
    PFNGLPOINTPARAMETERFPROC                        glPointParameterf;
    PFNGLPOINTPARAMETERFVPROC                       glPointParameterfv;
    PFNGLPOINTPARAMETERIPROC                        glPointParameteri;
    PFNGLPOINTPARAMETERIVPROC                       glPointParameteriv;
                                            
    // version 1.5 ////////////////////////////////////////////////////////////////////////////////
    PFNGLGENQUERIESPROC                             glGenQueries;
    PFNGLDELETEQUERIESPROC                          glDeleteQueries;
    PFNGLISQUERYPROC                                glIsQuery;
    PFNGLBEGINQUERYPROC                             glBeginQuery;
    PFNGLENDQUERYPROC                               glEndQuery;
    PFNGLGETQUERYIVPROC                             glGetQueryiv;
    PFNGLGETQUERYOBJECTIVPROC                       glGetQueryObjectiv;
    PFNGLGETQUERYOBJECTUIVPROC                      glGetQueryObjectuiv;
    PFNGLBINDBUFFERPROC                             glBindBuffer;
    PFNGLDELETEBUFFERSPROC                          glDeleteBuffers;
    PFNGLGENBUFFERSPROC                             glGenBuffers;
    PFNGLISBUFFERPROC                               glIsBuffer;
    PFNGLBUFFERDATAPROC                             glBufferData;
    PFNGLBUFFERSUBDATAPROC                          glBufferSubData;
    PFNGLGETBUFFERSUBDATAPROC                       glGetBufferSubData;
    PFNGLMAPBUFFERPROC                              glMapBuffer;
    PFNGLUNMAPBUFFERPROC                            glUnmapBuffer;
    PFNGLGETBUFFERPARAMETERIVPROC                   glGetBufferParameteriv;
    PFNGLGETBUFFERPOINTERVPROC                      glGetBufferPointerv;

    // version 2.0 ////////////////////////////////////////////////////////////////////////////////
    PFNGLBLENDEQUATIONSEPARATEPROC                  glBlendEquationSeparate;
    PFNGLDRAWBUFFERSPROC                            glDrawBuffers;
    PFNGLSTENCILOPSEPARATEPROC                      glStencilOpSeparate;
    PFNGLSTENCILFUNCSEPARATEPROC                    glStencilFuncSeparate;
    PFNGLSTENCILMASKSEPARATEPROC                    glStencilMaskSeparate;
    PFNGLATTACHSHADERPROC                           glAttachShader;
    PFNGLBINDATTRIBLOCATIONPROC                     glBindAttribLocation;
    PFNGLCOMPILESHADERPROC                          glCompileShader;
    PFNGLCREATEPROGRAMPROC                          glCreateProgram;
    PFNGLCREATESHADERPROC                           glCreateShader;
    PFNGLDELETEPROGRAMPROC                          glDeleteProgram;
    PFNGLDELETESHADERPROC                           glDeleteShader;
    PFNGLDETACHSHADERPROC                           glDetachShader;
    PFNGLDISABLEVERTEXATTRIBARRAYPROC               glDisableVertexAttribArray;
    PFNGLENABLEVERTEXATTRIBARRAYPROC                glEnableVertexAttribArray;
    PFNGLGETACTIVEATTRIBPROC                        glGetActiveAttrib;
    PFNGLGETACTIVEUNIFORMPROC                       glGetActiveUniform;
    PFNGLGETATTACHEDSHADERSPROC                     glGetAttachedShaders;
    PFNGLGETATTRIBLOCATIONPROC                      glGetAttribLocation;
    PFNGLGETPROGRAMIVPROC                           glGetProgramiv;
    PFNGLGETPROGRAMINFOLOGPROC                      glGetProgramInfoLog;
    PFNGLGETSHADERIVPROC                            glGetShaderiv;
    PFNGLGETSHADERINFOLOGPROC                       glGetShaderInfoLog;
    PFNGLGETSHADERSOURCEPROC                        glGetShaderSource;
    PFNGLGETUNIFORMLOCATIONPROC                     glGetUniformLocation;
    PFNGLGETUNIFORMFVPROC                           glGetUniformfv;
    PFNGLGETUNIFORMIVPROC                           glGetUniformiv;
    PFNGLGETVERTEXATTRIBDVPROC                      glGetVertexAttribdv;
    PFNGLGETVERTEXATTRIBFVPROC                      glGetVertexAttribfv;
    PFNGLGETVERTEXATTRIBIVPROC                      glGetVertexAttribiv;
    PFNGLGETVERTEXATTRIBPOINTERVPROC                glGetVertexAttribPointerv;
    PFNGLISPROGRAMPROC                              glIsProgram;
    PFNGLISSHADERPROC                               glIsShader;
    PFNGLLINKPROGRAMPROC                            glLinkProgram;
    PFNGLSHADERSOURCEPROC                           glShaderSource;
    PFNGLUSEPROGRAMPROC                             glUseProgram;
    PFNGLUNIFORM1FPROC                              glUniform1f;
    PFNGLUNIFORM2FPROC                              glUniform2f;
    PFNGLUNIFORM3FPROC                              glUniform3f;
    PFNGLUNIFORM4FPROC                              glUniform4f;
    PFNGLUNIFORM1IPROC                              glUniform1i;
    PFNGLUNIFORM2IPROC                              glUniform2i;
    PFNGLUNIFORM3IPROC                              glUniform3i;
    PFNGLUNIFORM4IPROC                              glUniform4i;
    PFNGLUNIFORM1FVPROC                             glUniform1fv;
    PFNGLUNIFORM2FVPROC                             glUniform2fv;
    PFNGLUNIFORM3FVPROC                             glUniform3fv;
    PFNGLUNIFORM4FVPROC                             glUniform4fv;
    PFNGLUNIFORM1IVPROC                             glUniform1iv;
    PFNGLUNIFORM2IVPROC                             glUniform2iv;
    PFNGLUNIFORM3IVPROC                             glUniform3iv;
    PFNGLUNIFORM4IVPROC                             glUniform4iv;
    PFNGLUNIFORMMATRIX2FVPROC                       glUniformMatrix2fv;
    PFNGLUNIFORMMATRIX3FVPROC                       glUniformMatrix3fv;
    PFNGLUNIFORMMATRIX4FVPROC                       glUniformMatrix4fv;
    PFNGLVALIDATEPROGRAMPROC                        glValidateProgram;
    PFNGLVERTEXATTRIB1DPROC                         glVertexAttrib1d;
    PFNGLVERTEXATTRIB1DVPROC                        glVertexAttrib1dv;
    PFNGLVERTEXATTRIB1FPROC                         glVertexAttrib1f;
    PFNGLVERTEXATTRIB1FVPROC                        glVertexAttrib1fv;
    PFNGLVERTEXATTRIB1SPROC                         glVertexAttrib1s;
    PFNGLVERTEXATTRIB1SVPROC                        glVertexAttrib1sv;
    PFNGLVERTEXATTRIB2DPROC                         glVertexAttrib2d;
    PFNGLVERTEXATTRIB2DVPROC                        glVertexAttrib2dv;
    PFNGLVERTEXATTRIB2FPROC                         glVertexAttrib2f;
    PFNGLVERTEXATTRIB2FVPROC                        glVertexAttrib2fv;
    PFNGLVERTEXATTRIB2SPROC                         glVertexAttrib2s;
    PFNGLVERTEXATTRIB2SVPROC                        glVertexAttrib2sv;
    PFNGLVERTEXATTRIB3DPROC                         glVertexAttrib3d;
    PFNGLVERTEXATTRIB3DVPROC                        glVertexAttrib3dv;
    PFNGLVERTEXATTRIB3FPROC                         glVertexAttrib3f;
    PFNGLVERTEXATTRIB3FVPROC                        glVertexAttrib3fv;
    PFNGLVERTEXATTRIB3SPROC                         glVertexAttrib3s;
    PFNGLVERTEXATTRIB3SVPROC                        glVertexAttrib3sv;
    PFNGLVERTEXATTRIB4NBVPROC                       glVertexAttrib4Nbv;
    PFNGLVERTEXATTRIB4NIVPROC                       glVertexAttrib4Niv;
    PFNGLVERTEXATTRIB4NSVPROC                       glVertexAttrib4Nsv;
    PFNGLVERTEXATTRIB4NUBPROC                       glVertexAttrib4Nub;
    PFNGLVERTEXATTRIB4NUBVPROC                      glVertexAttrib4Nubv;
    PFNGLVERTEXATTRIB4NUIVPROC                      glVertexAttrib4Nuiv;
    PFNGLVERTEXATTRIB4NUSVPROC                      glVertexAttrib4Nusv;
    PFNGLVERTEXATTRIB4BVPROC                        glVertexAttrib4bv;
    PFNGLVERTEXATTRIB4DPROC                         glVertexAttrib4d;
    PFNGLVERTEXATTRIB4DVPROC                        glVertexAttrib4dv;
    PFNGLVERTEXATTRIB4FPROC                         glVertexAttrib4f;
    PFNGLVERTEXATTRIB4FVPROC                        glVertexAttrib4fv;
    PFNGLVERTEXATTRIB4IVPROC                        glVertexAttrib4iv;
    PFNGLVERTEXATTRIB4SPROC                         glVertexAttrib4s;
    PFNGLVERTEXATTRIB4SVPROC                        glVertexAttrib4sv;
    PFNGLVERTEXATTRIB4UBVPROC                       glVertexAttrib4ubv;
    PFNGLVERTEXATTRIB4UIVPROC                       glVertexAttrib4uiv;
    PFNGLVERTEXATTRIB4USVPROC                       glVertexAttrib4usv;
    PFNGLVERTEXATTRIBPOINTERPROC                    glVertexAttribPointer;
                                            
    // version 2.1 ////////////////////////////////////////////////////////////////////////////////
    PFNGLUNIFORMMATRIX2X3FVPROC                     glUniformMatrix2x3fv;
    PFNGLUNIFORMMATRIX3X2FVPROC                     glUniformMatrix3x2fv;
    PFNGLUNIFORMMATRIX2X4FVPROC                     glUniformMatrix2x4fv;
    PFNGLUNIFORMMATRIX4X2FVPROC                     glUniformMatrix4x2fv;
    PFNGLUNIFORMMATRIX3X4FVPROC                     glUniformMatrix3x4fv;
    PFNGLUNIFORMMATRIX4X3FVPROC                     glUniformMatrix4x3fv;

    // version 3.0 ////////////////////////////////////////////////////////////////////////////////
    PFNGLCOLORMASKIPROC                             glColorMaski;
    PFNGLGETBOOLEANI_VPROC                          glGetBooleani_v;
    PFNGLGETINTEGERI_VPROC                          glGetIntegeri_v;
    PFNGLENABLEIPROC                                glEnablei;
    PFNGLDISABLEIPROC                               glDisablei;
    PFNGLISENABLEDIPROC                             glIsEnabledi;
    PFNGLBEGINTRANSFORMFEEDBACKPROC                 glBeginTransformFeedback;
    PFNGLENDTRANSFORMFEEDBACKPROC                   glEndTransformFeedback;
    PFNGLBINDBUFFERRANGEPROC                        glBindBufferRange;
    PFNGLBINDBUFFERBASEPROC                         glBindBufferBase;
    PFNGLTRANSFORMFEEDBACKVARYINGSPROC              glTransformFeedbackVaryings;
    PFNGLGETTRANSFORMFEEDBACKVARYINGPROC            glGetTransformFeedbackVarying;
    PFNGLCLAMPCOLORPROC                             glClampColor;
    PFNGLBEGINCONDITIONALRENDERPROC                 glBeginConditionalRender;
    PFNGLENDCONDITIONALRENDERPROC                   glEndConditionalRender;
    PFNGLVERTEXATTRIBIPOINTERPROC                   glVertexAttribIPointer;
    PFNGLGETVERTEXATTRIBIIVPROC                     glGetVertexAttribIiv;
    PFNGLGETVERTEXATTRIBIUIVPROC                    glGetVertexAttribIuiv;
    PFNGLVERTEXATTRIBI1IPROC                        glVertexAttribI1i;
    PFNGLVERTEXATTRIBI2IPROC                        glVertexAttribI2i;
    PFNGLVERTEXATTRIBI3IPROC                        glVertexAttribI3i;
    PFNGLVERTEXATTRIBI4IPROC                        glVertexAttribI4i;
    PFNGLVERTEXATTRIBI1UIPROC                       glVertexAttribI1ui;
    PFNGLVERTEXATTRIBI2UIPROC                       glVertexAttribI2ui;
    PFNGLVERTEXATTRIBI3UIPROC                       glVertexAttribI3ui;
    PFNGLVERTEXATTRIBI4UIPROC                       glVertexAttribI4ui;
    PFNGLVERTEXATTRIBI1IVPROC                       glVertexAttribI1iv;
    PFNGLVERTEXATTRIBI2IVPROC                       glVertexAttribI2iv;
    PFNGLVERTEXATTRIBI3IVPROC                       glVertexAttribI3iv;
    PFNGLVERTEXATTRIBI4IVPROC                       glVertexAttribI4iv;
    PFNGLVERTEXATTRIBI1UIVPROC                      glVertexAttribI1uiv;
    PFNGLVERTEXATTRIBI2UIVPROC                      glVertexAttribI2uiv;
    PFNGLVERTEXATTRIBI3UIVPROC                      glVertexAttribI3uiv;
    PFNGLVERTEXATTRIBI4UIVPROC                      glVertexAttribI4uiv;
    PFNGLVERTEXATTRIBI4BVPROC                       glVertexAttribI4bv;
    PFNGLVERTEXATTRIBI4SVPROC                       glVertexAttribI4sv;
    PFNGLVERTEXATTRIBI4UBVPROC                      glVertexAttribI4ubv;
    PFNGLVERTEXATTRIBI4USVPROC                      glVertexAttribI4usv;
    PFNGLGETUNIFORMUIVPROC                          glGetUniformuiv;
    PFNGLBINDFRAGDATALOCATIONPROC                   glBindFragDataLocation;
    PFNGLGETFRAGDATALOCATIONPROC                    glGetFragDataLocation;
    PFNGLUNIFORM1UIPROC                             glUniform1ui;
    PFNGLUNIFORM2UIPROC                             glUniform2ui;
    PFNGLUNIFORM3UIPROC                             glUniform3ui;
    PFNGLUNIFORM4UIPROC                             glUniform4ui;
    PFNGLUNIFORM1UIVPROC                            glUniform1uiv;
    PFNGLUNIFORM2UIVPROC                            glUniform2uiv;
    PFNGLUNIFORM3UIVPROC                            glUniform3uiv;
    PFNGLUNIFORM4UIVPROC                            glUniform4uiv;
    PFNGLTEXPARAMETERIIVPROC                        glTexParameterIiv;
    PFNGLTEXPARAMETERIUIVPROC                       glTexParameterIuiv;
    PFNGLGETTEXPARAMETERIIVPROC                     glGetTexParameterIiv;
    PFNGLGETTEXPARAMETERIUIVPROC                    glGetTexParameterIuiv;
    PFNGLCLEARBUFFERIVPROC                          glClearBufferiv;
    PFNGLCLEARBUFFERUIVPROC                         glClearBufferuiv;
    PFNGLCLEARBUFFERFVPROC                          glClearBufferfv;
    PFNGLCLEARBUFFERFIPROC                          glClearBufferfi;
    PFNGLGETSTRINGIPROC                             glGetStringi;
    // use ARB_framebuffer_object
    PFNGLISRENDERBUFFERPROC                         glIsRenderbuffer;
    PFNGLBINDRENDERBUFFERPROC                       glBindRenderbuffer;
    PFNGLDELETERENDERBUFFERSPROC                    glDeleteRenderbuffers;
    PFNGLGENRENDERBUFFERSPROC                       glGenRenderbuffers;
    PFNGLRENDERBUFFERSTORAGEPROC                    glRenderbufferStorage;
    PFNGLGETRENDERBUFFERPARAMETERIVPROC             glGetRenderbufferParameteriv;
    PFNGLISFRAMEBUFFERPROC                          glIsFramebuffer;
    PFNGLBINDFRAMEBUFFERPROC                        glBindFramebuffer;
    PFNGLDELETEFRAMEBUFFERSPROC                     glDeleteFramebuffers;
    PFNGLGENFRAMEBUFFERSPROC                        glGenFramebuffers;
    PFNGLCHECKFRAMEBUFFERSTATUSPROC                 glCheckFramebufferStatus;
    PFNGLFRAMEBUFFERTEXTURE1DPROC                   glFramebufferTexture1D;
    PFNGLFRAMEBUFFERTEXTURE2DPROC                   glFramebufferTexture2D;
    PFNGLFRAMEBUFFERTEXTURE3DPROC                   glFramebufferTexture3D;
    PFNGLFRAMEBUFFERRENDERBUFFERPROC                glFramebufferRenderbuffer;
    PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC    glGetFramebufferAttachmentParameteriv;
    PFNGLGENERATEMIPMAPPROC                         glGenerateMipmap;
    PFNGLBLITFRAMEBUFFERPROC                        glBlitFramebuffer;
    PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC         glRenderbufferStorageMultisample;
    PFNGLFRAMEBUFFERTEXTURELAYERPROC                glFramebufferTextureLayer;
    // use ARB_map_buffer_ranger
    PFNGLMAPBUFFERRANGEPROC                         glMapBufferRange;
    PFNGLFLUSHMAPPEDBUFFERRANGEPROC                 glFlushMappedBufferRange;
    // use ARB_vertex_array_object          
    PFNGLBINDVERTEXARRAYPROC                        glBindVertexArray;
    PFNGLDELETEVERTEXARRAYSPROC                     glDeleteVertexArrays;
    PFNGLGENVERTEXARRAYSPROC                        glGenVertexArrays;
    PFNGLISVERTEXARRAYPROC                          glIsVertexArray;
                                            
    // version 3.1 ////////////////////////////////////////////////////////////////////////////////
    PFNGLDRAWARRAYSINSTANCEDPROC                    glDrawArraysInstanced;
    PFNGLDRAWELEMENTSINSTANCEDPROC                  glDrawElementsInstanced;
    PFNGLTEXBUFFERPROC                              glTexBuffer;
    PFNGLPRIMITIVERESTARTINDEXPROC                  glPrimitiveRestartIndex;
    // use ARB_copy_buffer                  
    PFNGLCOPYBUFFERSUBDATAPROC                      glCopyBufferSubData;
    // use ARB_uniform_buffer_object        
    PFNGLGETUNIFORMINDICESPROC                      glGetUniformIndices;
    PFNGLGETACTIVEUNIFORMSIVPROC                    glGetActiveUniformsiv;
    PFNGLGETACTIVEUNIFORMNAMEPROC                   glGetActiveUniformName;
    PFNGLGETUNIFORMBLOCKINDEXPROC                   glGetUniformBlockIndex;
    PFNGLGETACTIVEUNIFORMBLOCKIVPROC                glGetActiveUniformBlockiv;
    PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC              glGetActiveUniformBlockName;
    PFNGLUNIFORMBLOCKBINDINGPROC                    glUniformBlockBinding;
                                            
    // version 3.2 ////////////////////////////////////////////////////////////////////////////////
    PFNGLGETINTEGER64I_VPROC                        glGetInteger64i_v;
    PFNGLGETBUFFERPARAMETERI64VPROC                 glGetBufferParameteri64v;
    //PFNGLPROGRAMPARAMETERIPROC                      glProgramParameteri;
    PFNGLFRAMEBUFFERTEXTUREPROC                     glFramebufferTexture;
    //PFNGLFRAMEBUFFERTEXTUREFACEPROC                 glFramebufferTextureFace;
    // use ARB_draw_elements_base_vertex    
    PFNGLDRAWELEMENTSBASEVERTEXPROC                 glDrawElementsBaseVertex;
    PFNGLDRAWRANGEELEMENTSBASEVERTEXPROC            glDrawRangeElementsBaseVertex;
    PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC        glDrawElementsInstancedBaseVertex;
    PFNGLMULTIDRAWELEMENTSBASEVERTEXPROC            glMultiDrawElementsBaseVertex;
    // use ARB_provoking_vertex             
    PFNGLPROVOKINGVERTEXPROC                        glProvokingVertex;
    // use ARB_sync                         
    PFNGLFENCESYNCPROC                              glFenceSync;
    PFNGLISSYNCPROC                                 glIsSync;
    PFNGLDELETESYNCPROC                             glDeleteSync;
    PFNGLCLIENTWAITSYNCPROC                         glClientWaitSync;
    PFNGLWAITSYNCPROC                               glWaitSync;
    PFNGLGETINTEGER64VPROC                          glGetInteger64v;
    PFNGLGETSYNCIVPROC                              glGetSynciv;
    // use ARB_texture_multisample          
    PFNGLTEXIMAGE2DMULTISAMPLEPROC                  glTexImage2DMultisample;
    PFNGLTEXIMAGE3DMULTISAMPLEPROC                  glTexImage3DMultisample;
    PFNGLGETMULTISAMPLEFVPROC                       glGetMultisamplefv;
    PFNGLSAMPLEMASKIPROC                            glSampleMaski;

    // version 3.3 ////////////////////////////////////////////////////////////////////////////////
    // use GL_ARB_shader_bit_encoding
    // use ARB_blend_func_extended
    PFNGLBINDFRAGDATALOCATIONINDEXEDPROC            glBindFragDataLocationIndexed;
    PFNGLGETFRAGDATAINDEXPROC                       glGetFragDataIndex;
    // use GL_ARB_explicit_attrib_location
    // use GL_ARB_occlusion_query2
    // use ARB_sampler_objects
    PFNGLGENSAMPLERSPROC                            glGenSamplers;
    PFNGLDELETESAMPLERSPROC                         glDeleteSamplers;
    PFNGLISSAMPLERPROC                              glIsSampler;
    PFNGLBINDSAMPLERPROC                            glBindSampler;
    PFNGLSAMPLERPARAMETERIPROC                      glSamplerParameteri;
    PFNGLSAMPLERPARAMETERIVPROC                     glSamplerParameteriv;
    PFNGLSAMPLERPARAMETERFPROC                      glSamplerParameterf;
    PFNGLSAMPLERPARAMETERFVPROC                     glSamplerParameterfv;
    PFNGLSAMPLERPARAMETERIIVPROC                    glSamplerParameterIiv;
    PFNGLSAMPLERPARAMETERIUIVPROC                   glSamplerParameterIuiv;
    PFNGLGETSAMPLERPARAMETERIVPROC                  glGetSamplerParameteriv;
    PFNGLGETSAMPLERPARAMETERIIVPROC                 glGetSamplerParameterIiv;
    PFNGLGETSAMPLERPARAMETERFVPROC                  glGetSamplerParameterfv;
    //PFNGLGETSAMPLERPARAMETERIFVPROC                 glGetSamplerParameterIfv;
    // use GL_ARB_texture_rgb10_a2ui
    // use GL_ARB_texture_swizzle
    // use ARB_timer_query
    PFNGLQUERYCOUNTERPROC                           glQueryCounter;
    PFNGLGETQUERYOBJECTI64VPROC                     glGetQueryObjecti64v;
    PFNGLGETQUERYOBJECTUI64VPROC                    glGetQueryObjectui64v;
    // use GL_ARB_texture_swizzle
    /// TODO missing entry points
    // use ARB_vertex_type_2_10_10_10_rev
    // non which concern core profile

    // version 4.0 ////////////////////////////////////////////////////////////////////////////////
    // use ARB_draw_buffers_blend
    PFNGLBLENDEQUATIONIPROC                         glBlendEquationi;
    PFNGLBLENDEQUATIONSEPARATEIPROC                 glBlendEquationSeparatei;
    PFNGLBLENDFUNCIPROC                             glBlendFunci;
    PFNGLBLENDFUNCSEPARATEIPROC                     glBlendFuncSeparatei;
    // use ARB_draw_indirect
    PFNGLDRAWARRAYSINDIRECTPROC                     glDrawArraysIndirect;
    PFNGLDRAWELEMENTSINDIRECTPROC                   glDrawElementsIndirect;
    // use ARB_gpu_shader5
    // use ARB_gpu_shader_fp64
    PFNGLUNIFORM1DPROC                              glUniform1d;
    PFNGLUNIFORM2DPROC                              glUniform2d;
    PFNGLUNIFORM3DPROC                              glUniform3d;
    PFNGLUNIFORM4DPROC                              glUniform4d;
    PFNGLUNIFORM1DVPROC                             glUniform1dv;
    PFNGLUNIFORM2DVPROC                             glUniform2dv;
    PFNGLUNIFORM3DVPROC                             glUniform3dv;
    PFNGLUNIFORM4DVPROC                             glUniform4dv;
    PFNGLUNIFORMMATRIX2DVPROC                       glUniformMatrix2dv;
    PFNGLUNIFORMMATRIX3DVPROC                       glUniformMatrix3dv;
    PFNGLUNIFORMMATRIX4DVPROC                       glUniformMatrix4dv;
    PFNGLUNIFORMMATRIX2X3DVPROC                     glUniformMatrix2x3dv;
    PFNGLUNIFORMMATRIX2X4DVPROC                     glUniformMatrix2x4dv;
    PFNGLUNIFORMMATRIX3X2DVPROC                     glUniformMatrix3x2dv;
    PFNGLUNIFORMMATRIX3X4DVPROC                     glUniformMatrix3x4dv;
    PFNGLUNIFORMMATRIX4X2DVPROC                     glUniformMatrix4x2dv;
    PFNGLUNIFORMMATRIX4X3DVPROC                     glUniformMatrix4x3dv;
    PFNGLGETUNIFORMDVPROC                           glGetUniformdv;
    PFNGLPROGRAMUNIFORM1DEXTPROC                    glProgramUniform1dEXT;
    PFNGLPROGRAMUNIFORM2DEXTPROC                    glProgramUniform2dEXT;
    PFNGLPROGRAMUNIFORM3DEXTPROC                    glProgramUniform3dEXT;
    PFNGLPROGRAMUNIFORM4DEXTPROC                    glProgramUniform4dEXT;
    PFNGLPROGRAMUNIFORM1DVEXTPROC                   glProgramUniform1dvEXT;
    PFNGLPROGRAMUNIFORM2DVEXTPROC                   glProgramUniform2dvEXT;
    PFNGLPROGRAMUNIFORM3DVEXTPROC                   glProgramUniform3dvEXT;
    PFNGLPROGRAMUNIFORM4DVEXTPROC                   glProgramUniform4dvEXT;
    PFNGLPROGRAMUNIFORMMATRIX2DVEXTPROC             glProgramUniformMatrix2dvEXT;
    PFNGLPROGRAMUNIFORMMATRIX3DVEXTPROC             glProgramUniformMatrix3dvEXT;
    PFNGLPROGRAMUNIFORMMATRIX4DVEXTPROC             glProgramUniformMatrix4dvEXT;
    PFNGLPROGRAMUNIFORMMATRIX2X3DVEXTPROC           glProgramUniformMatrix2x3dvEXT;
    PFNGLPROGRAMUNIFORMMATRIX2X4DVEXTPROC           glProgramUniformMatrix2x4dvEXT;
    PFNGLPROGRAMUNIFORMMATRIX3X2DVEXTPROC           glProgramUniformMatrix3x2dvEXT;
    PFNGLPROGRAMUNIFORMMATRIX3X4DVEXTPROC           glProgramUniformMatrix3x4dvEXT;
    PFNGLPROGRAMUNIFORMMATRIX4X2DVEXTPROC           glProgramUniformMatrix4x2dvEXT;
    PFNGLPROGRAMUNIFORMMATRIX4X3DVEXTPROC           glProgramUniformMatrix4x3dvEXT;
    // use ARB_sample_shading
    PFNGLMINSAMPLESHADINGPROC                       glMinSampleShading;
    // use ARB_shader_subroutine
    PFNGLGETSUBROUTINEUNIFORMLOCATIONPROC           glGetSubroutineUniformLocation;
    PFNGLGETSUBROUTINEINDEXPROC                     glGetSubroutineIndex;
    PFNGLGETACTIVESUBROUTINEUNIFORMIVPROC           glGetActiveSubroutineUniformiv;
    PFNGLGETACTIVESUBROUTINEUNIFORMNAMEPROC         glGetActiveSubroutineUniformName;
    PFNGLGETACTIVESUBROUTINENAMEPROC                glGetActiveSubroutineName;
    PFNGLUNIFORMSUBROUTINESUIVPROC                  glUniformSubroutinesuiv;
    PFNGLGETUNIFORMSUBROUTINEUIVPROC                glGetUniformSubroutineuiv;
    PFNGLGETPROGRAMSTAGEIVPROC                      glGetProgramStageiv;
    // use ARB_tessellation_shader
    PFNGLPATCHPARAMETERIPROC                        glPatchParameteri;
    PFNGLPATCHPARAMETERFVPROC                       glPatchParameterfv;
    // use ARB_texture_buffer_object_rgb32
    // use ARB_texture_cube_map_array
    // use ARB_transform_feedback2
    PFNGLBINDTRANSFORMFEEDBACKPROC                  glBindTransformFeedback;
    PFNGLDELETETRANSFORMFEEDBACKSPROC               glDeleteTransformFeedbacks;
    PFNGLGENTRANSFORMFEEDBACKSPROC                  glGenTransformFeedbacks;
    PFNGLISTRANSFORMFEEDBACKPROC                    glIsTransformFeedback;
    PFNGLPAUSETRANSFORMFEEDBACKPROC                 glPauseTransformFeedback;
    PFNGLRESUMETRANSFORMFEEDBACKPROC                glResumeTransformFeedback;
    PFNGLDRAWTRANSFORMFEEDBACKPROC                  glDrawTransformFeedback;
    // use ARB_transform_feedback3
    PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC            glDrawTransformFeedbackStream;
    PFNGLBEGINQUERYINDEXEDPROC                      glBeginQueryIndexed;
    PFNGLENDQUERYINDEXEDPROC                        glEndQueryIndexed;
    PFNGLGETQUERYINDEXEDIVPROC                      glGetQueryIndexediv;

    // version 4.1 ////////////////////////////////////////////////////////////////////////////////
    // use  ARB_ES2_compatibility
    PFNGLRELEASESHADERCOMPILERPROC                  glReleaseShaderCompiler;
    PFNGLSHADERBINARYPROC                           glShaderBinary;
    PFNGLGETSHADERPRECISIONFORMATPROC               glGetShaderPrecisionFormat;
    PFNGLDEPTHRANGEFPROC                            glDepthRangef;
    PFNGLCLEARDEPTHFPROC                            glClearDepthf;
    // use ARB_get_program_binary
    PFNGLGETPROGRAMBINARYPROC                       glGetProgramBinary;
    PFNGLPROGRAMBINARYPROC                          glProgramBinary;
    PFNGLPROGRAMPARAMETERIPROC                      glProgramParameteri;
    // use ARB_separate_shader_objects
    PFNGLUSEPROGRAMSTAGESPROC                       glUseProgramStages;
    PFNGLACTIVESHADERPROGRAMPROC                    glActiveShaderProgram;
    PFNGLCREATESHADERPROGRAMVPROC                   glCreateShaderProgramv;
    PFNGLBINDPROGRAMPIPELINEPROC                    glBindProgramPipeline;
    PFNGLDELETEPROGRAMPIPELINESPROC                 glDeleteProgramPipelines;
    PFNGLGENPROGRAMPIPELINESPROC                    glGenProgramPipelines;
    PFNGLISPROGRAMPIPELINEPROC                      glIsProgramPipeline;
    PFNGLGETPROGRAMPIPELINEIVPROC                   glGetProgramPipelineiv;
    PFNGLPROGRAMUNIFORM1IPROC                       glProgramUniform1i;
    PFNGLPROGRAMUNIFORM1IVPROC                      glProgramUniform1iv;
    PFNGLPROGRAMUNIFORM1FPROC                       glProgramUniform1f;
    PFNGLPROGRAMUNIFORM1FVPROC                      glProgramUniform1fv;
    PFNGLPROGRAMUNIFORM1DPROC                       glProgramUniform1d;
    PFNGLPROGRAMUNIFORM1DVPROC                      glProgramUniform1dv;
    PFNGLPROGRAMUNIFORM1UIPROC                      glProgramUniform1ui;
    PFNGLPROGRAMUNIFORM1UIVPROC                     glProgramUniform1uiv;
    PFNGLPROGRAMUNIFORM2IPROC                       glProgramUniform2i;
    PFNGLPROGRAMUNIFORM2IVPROC                      glProgramUniform2iv;
    PFNGLPROGRAMUNIFORM2FPROC                       glProgramUniform2f;
    PFNGLPROGRAMUNIFORM2FVPROC                      glProgramUniform2fv;
    PFNGLPROGRAMUNIFORM2DPROC                       glProgramUniform2d;
    PFNGLPROGRAMUNIFORM2DVPROC                      glProgramUniform2dv;
    PFNGLPROGRAMUNIFORM2UIPROC                      glProgramUniform2ui;
    PFNGLPROGRAMUNIFORM2UIVPROC                     glProgramUniform2uiv;
    PFNGLPROGRAMUNIFORM3IPROC                       glProgramUniform3i;
    PFNGLPROGRAMUNIFORM3IVPROC                      glProgramUniform3iv;
    PFNGLPROGRAMUNIFORM3FPROC                       glProgramUniform3f;
    PFNGLPROGRAMUNIFORM3FVPROC                      glProgramUniform3fv;
    PFNGLPROGRAMUNIFORM3DPROC                       glProgramUniform3d;
    PFNGLPROGRAMUNIFORM3DVPROC                      glProgramUniform3dv;
    PFNGLPROGRAMUNIFORM3UIPROC                      glProgramUniform3ui;
    PFNGLPROGRAMUNIFORM3UIVPROC                     glProgramUniform3uiv;
    PFNGLPROGRAMUNIFORM4IPROC                       glProgramUniform4i;
    PFNGLPROGRAMUNIFORM4IVPROC                      glProgramUniform4iv;
    PFNGLPROGRAMUNIFORM4FPROC                       glProgramUniform4f;
    PFNGLPROGRAMUNIFORM4FVPROC                      glProgramUniform4fv;
    PFNGLPROGRAMUNIFORM4DPROC                       glProgramUniform4d;
    PFNGLPROGRAMUNIFORM4DVPROC                      glProgramUniform4dv;
    PFNGLPROGRAMUNIFORM4UIPROC                      glProgramUniform4ui;
    PFNGLPROGRAMUNIFORM4UIVPROC                     glProgramUniform4uiv;
    PFNGLPROGRAMUNIFORMMATRIX2FVPROC                glProgramUniformMatrix2fv;
    PFNGLPROGRAMUNIFORMMATRIX3FVPROC                glProgramUniformMatrix3fv;
    PFNGLPROGRAMUNIFORMMATRIX4FVPROC                glProgramUniformMatrix4fv;
    PFNGLPROGRAMUNIFORMMATRIX2DVPROC                glProgramUniformMatrix2dv;
    PFNGLPROGRAMUNIFORMMATRIX3DVPROC                glProgramUniformMatrix3dv;
    PFNGLPROGRAMUNIFORMMATRIX4DVPROC                glProgramUniformMatrix4dv;
    PFNGLPROGRAMUNIFORMMATRIX2X3FVPROC              glProgramUniformMatrix2x3fv;
    PFNGLPROGRAMUNIFORMMATRIX3X2FVPROC              glProgramUniformMatrix3x2fv;
    PFNGLPROGRAMUNIFORMMATRIX2X4FVPROC              glProgramUniformMatrix2x4fv;
    PFNGLPROGRAMUNIFORMMATRIX4X2FVPROC              glProgramUniformMatrix4x2fv;
    PFNGLPROGRAMUNIFORMMATRIX3X4FVPROC              glProgramUniformMatrix3x4fv;
    PFNGLPROGRAMUNIFORMMATRIX4X3FVPROC              glProgramUniformMatrix4x3fv;
    PFNGLPROGRAMUNIFORMMATRIX2X3DVPROC              glProgramUniformMatrix2x3dv;
    PFNGLPROGRAMUNIFORMMATRIX3X2DVPROC              glProgramUniformMatrix3x2dv;
    PFNGLPROGRAMUNIFORMMATRIX2X4DVPROC              glProgramUniformMatrix2x4dv;
    PFNGLPROGRAMUNIFORMMATRIX4X2DVPROC              glProgramUniformMatrix4x2dv;
    PFNGLPROGRAMUNIFORMMATRIX3X4DVPROC              glProgramUniformMatrix3x4dv;
    PFNGLPROGRAMUNIFORMMATRIX4X3DVPROC              glProgramUniformMatrix4x3dv;
    PFNGLVALIDATEPROGRAMPIPELINEPROC                glValidateProgramPipeline;
    PFNGLGETPROGRAMPIPELINEINFOLOGPROC              glGetProgramPipelineInfoLog;
    // use ARB_shader_precision (no entry points)
    // use ARB_vertex_attrib_64bit
    PFNGLVERTEXATTRIBL1DPROC                        glVertexAttribL1d;
    PFNGLVERTEXATTRIBL2DPROC                        glVertexAttribL2d;
    PFNGLVERTEXATTRIBL3DPROC                        glVertexAttribL3d;
    PFNGLVERTEXATTRIBL4DPROC                        glVertexAttribL4d;
    PFNGLVERTEXATTRIBL1DVPROC                       glVertexAttribL1dv;
    PFNGLVERTEXATTRIBL2DVPROC                       glVertexAttribL2dv;
    PFNGLVERTEXATTRIBL3DVPROC                       glVertexAttribL3dv;
    PFNGLVERTEXATTRIBL4DVPROC                       glVertexAttribL4dv;
    PFNGLVERTEXATTRIBLPOINTERPROC                   glVertexAttribLPointer;
    PFNGLGETVERTEXATTRIBLDVPROC                     glGetVertexAttribLdv;
    // use ARB_viewport_array
    PFNGLVIEWPORTARRAYVPROC                         glViewportArrayv;
    PFNGLVIEWPORTINDEXEDFPROC                       glViewportIndexedf;
    PFNGLVIEWPORTINDEXEDFVPROC                      glViewportIndexedfv;
    PFNGLSCISSORARRAYVPROC                          glScissorArrayv;
    PFNGLSCISSORINDEXEDPROC                         glScissorIndexed;
    PFNGLSCISSORINDEXEDVPROC                        glScissorIndexedv;
    PFNGLDEPTHRANGEARRAYVPROC                       glDepthRangeArrayv;
    PFNGLDEPTHRANGEINDEXEDPROC                      glDepthRangeIndexed;
    PFNGLGETFLOATI_VPROC                            glGetFloati_v;
    PFNGLGETDOUBLEI_VPROC                           glGetDoublei_v;

    // version 4.2 ////////////////////////////////////////////////////////////////////////////////
    // ARB_base_instance
    PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC        glDrawArraysInstancedBaseInstance;
    PFNGLDRAWELEMENTSINSTANCEDBASEINSTANCEPROC      glDrawElementsInstancedBaseInstance;
    PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC glDrawElementsInstancedBaseVertexBaseInstance;
    // ARB_shading_language_420pack (no entry points)
    // ARB_transform_feedback_instanced
    PFNGLDRAWTRANSFORMFEEDBACKINSTANCEDPROC         glDrawTransformFeedbackInstanced;
    PFNGLDRAWTRANSFORMFEEDBACKSTREAMINSTANCEDPROC   glDrawTransformFeedbackStreamInstanced;
    // ARB_compressed_texture_pixel_storage (no entry points)
    // ARB_conservative_depth (no entry points)
    // ARB_internalformat_query
    PFNGLGETINTERNALFORMATIVPROC                    glGetInternalformativ;
    // ARB_map_buffer_alignment (no entry points)
    // ARB_shader_atomic_counters
    PFNGLGETACTIVEATOMICCOUNTERBUFFERIVPROC         glGetActiveAtomicCounterBufferiv;
    // ARB_shader_image_load_store
    PFNGLBINDIMAGETEXTUREPROC                       glBindImageTexture;
    PFNGLMEMORYBARRIERPROC                          glMemoryBarrier;
    // ARB_shading_language_packing (no entry points)
    // ARB_texture_storage
    PFNGLTEXSTORAGE1DPROC                           glTexStorage1D;
    PFNGLTEXSTORAGE2DPROC                           glTexStorage2D;
    PFNGLTEXSTORAGE3DPROC                           glTexStorage3D;
    PFNGLTEXTURESTORAGE1DEXTPROC                    glTextureStorage1DEXT;
    PFNGLTEXTURESTORAGE2DEXTPROC                    glTextureStorage2DEXT;
    PFNGLTEXTURESTORAGE3DEXTPROC                    glTextureStorage3DEXT;

    // version 4.3 ////////////////////////////////////////////////////////////////////////////////
    // ARB_arrays_of_arrays (no entry points, GLSL only)
    // ARB_fragment_layer_viewport (no entry points, GLSL only)
    // ARB_shader_image_size (no entry points, GLSL only)
    // ARB_ES3_compatibility (no entry points)
    // ARB_clear_buffer_object
    PFNGLCLEARBUFFERDATAPROC                        glClearBufferData;
    PFNGLCLEARBUFFERSUBDATAPROC                     glClearBufferSubData;
    PFNGLCLEARNAMEDBUFFERDATAEXTPROC                glClearNamedBufferDataEXT;
    PFNGLCLEARNAMEDBUFFERSUBDATAEXTPROC             glClearNamedBufferSubDataEXT;
    // ARB_compute_shader
    PFNGLDISPATCHCOMPUTEPROC                        glDispatchCompute;
    PFNGLDISPATCHCOMPUTEINDIRECTPROC                glDispatchComputeIndirect;
    // ARB_copy_image
    PFNGLCOPYIMAGESUBDATAPROC                       glCopyImageSubData;
    // KHR_debug (includes ARB_debug_output commands promoted to KHR without suffixes)
    PFNGLDEBUGMESSAGECONTROLPROC                    glDebugMessageControl;
    PFNGLDEBUGMESSAGEINSERTPROC                     glDebugMessageInsert;
    PFNGLDEBUGMESSAGECALLBACKPROC                   glDebugMessageCallback;
    PFNGLGETDEBUGMESSAGELOGPROC                     glGetDebugMessageLog;
    PFNGLPUSHDEBUGGROUPPROC                         glPushDebugGroup;
    PFNGLPOPDEBUGGROUPPROC                          glPopDebugGroup;
    PFNGLOBJECTLABELPROC                            glObjectLabel;
    PFNGLGETOBJECTLABELPROC                         glGetObjectLabel;
    PFNGLOBJECTPTRLABELPROC                         glObjectPtrLabel;
    PFNGLGETOBJECTPTRLABELPROC                      glGetObjectPtrLabel;
    // ARB_explicit_uniform_location (no entry points)
    // ARB_framebuffer_no_attachments
    PFNGLFRAMEBUFFERPARAMETERIPROC                  glFramebufferParameteri;
    PFNGLGETFRAMEBUFFERPARAMETERIVPROC              glGetFramebufferParameteriv;
    PFNGLNAMEDFRAMEBUFFERPARAMETERIEXTPROC          glNamedFramebufferParameteriEXT;
    PFNGLGETNAMEDFRAMEBUFFERPARAMETERIVEXTPROC      glGetNamedFramebufferParameterivEXT;
    // ARB_internalformat_query2
    PFNGLGETINTERNALFORMATI64VPROC                  glGetInternalformati64v;
    // ARB_invalidate_subdata
    PFNGLINVALIDATETEXSUBIMAGEPROC                  glInvalidateTexSubImage;
    PFNGLINVALIDATETEXIMAGEPROC                     glInvalidateTexImage;
    PFNGLINVALIDATEBUFFERSUBDATAPROC                glInvalidateBufferSubData;
    PFNGLINVALIDATEBUFFERDATAPROC                   glInvalidateBufferData;
    PFNGLINVALIDATEFRAMEBUFFERPROC                  glInvalidateFramebuffer;
    PFNGLINVALIDATESUBFRAMEBUFFERPROC               glInvalidateSubFramebuffer;
    // ARB_multi_draw_indirect
    PFNGLMULTIDRAWARRAYSINDIRECTPROC                glMultiDrawArraysIndirect;
    PFNGLMULTIDRAWELEMENTSINDIRECTPROC              glMultiDrawElementsIndirect;
    // ARB_program_interface_query
    PFNGLGETPROGRAMINTERFACEIVPROC                  glGetProgramInterfaceiv;
    PFNGLGETPROGRAMRESOURCEINDEXPROC                glGetProgramResourceIndex;
    PFNGLGETPROGRAMRESOURCENAMEPROC                 glGetProgramResourceName;
    PFNGLGETPROGRAMRESOURCEIVPROC                   glGetProgramResourceiv;
    PFNGLGETPROGRAMRESOURCELOCATIONPROC             glGetProgramResourceLocation;
    PFNGLGETPROGRAMRESOURCELOCATIONINDEXPROC        glGetProgramResourceLocationIndex;
    // ARB_robust_buffer_access_behavior (no entry points)
    // ARB_shader_storage_buffer_object
    PFNGLSHADERSTORAGEBLOCKBINDINGPROC              glShaderStorageBlockBinding;
    // ARB_stencil_texturing (no entry points)
    // ARB_texture_buffer_range
    PFNGLTEXBUFFERRANGEPROC                         glTexBufferRange;
    PFNGLTEXTUREBUFFERRANGEEXTPROC                  glTextureBufferRangeEXT;
    // ARB_texture_query_levels (no entry points)
    // ARB_texture_storage_multisample
    PFNGLTEXSTORAGE2DMULTISAMPLEPROC                glTexStorage2DMultisample;
    PFNGLTEXSTORAGE3DMULTISAMPLEPROC                glTexStorage3DMultisample;
    PFNGLTEXTURESTORAGE2DMULTISAMPLEEXTPROC         glTextureStorage2DMultisampleEXT;
    PFNGLTEXTURESTORAGE3DMULTISAMPLEEXTPROC         glTextureStorage3DMultisampleEXT;
    // ARB_texture_view
    PFNGLTEXTUREVIEWPROC                            glTextureView;
    // ARB_vertex_attrib_binding
    PFNGLBINDVERTEXBUFFERPROC                       glBindVertexBuffer;
    PFNGLVERTEXATTRIBFORMATPROC                     glVertexAttribFormat;
    PFNGLVERTEXATTRIBIFORMATPROC                    glVertexAttribIFormat;
    PFNGLVERTEXATTRIBLFORMATPROC                    glVertexAttribLFormat;
    PFNGLVERTEXATTRIBBINDINGPROC                    glVertexAttribBinding;
    PFNGLVERTEXBINDINGDIVISORPROC                   glVertexBindingDivisor;
    PFNGLVERTEXARRAYBINDVERTEXBUFFEREXTPROC         glVertexArrayBindVertexBufferEXT;
    PFNGLVERTEXARRAYVERTEXATTRIBFORMATEXTPROC       glVertexArrayVertexAttribFormatEXT;
    PFNGLVERTEXARRAYVERTEXATTRIBIFORMATEXTPROC      glVertexArrayVertexAttribIFormatEXT;
    PFNGLVERTEXARRAYVERTEXATTRIBLFORMATEXTPROC      glVertexArrayVertexAttribLFormatEXT;
    PFNGLVERTEXARRAYVERTEXATTRIBBINDINGEXTPROC      glVertexArrayVertexAttribBindingEXT;
    PFNGLVERTEXARRAYVERTEXBINDINGDIVISOREXTPROC     glVertexArrayVertexBindingDivisorEXT;

    // version 4.4 ////////////////////////////////////////////////////////////////////////////////
    // ARB_buffer_storage
    PFNGLBUFFERSTORAGEPROC                          glBufferStorage;
    // ARB_clear_texture
    PFNGLCLEARTEXIMAGEPROC                          glClearTexImage;
    PFNGLCLEARTEXSUBIMAGEPROC                       glClearTexSubImage;
    // ARB_enhanced_layouts (no entry points)
    // ARB_multi_bind
    PFNGLBINDBUFFERSBASEPROC                        glBindBuffersBase;
    PFNGLBINDBUFFERSRANGEPROC                       glBindBuffersRange; 
    PFNGLBINDTEXTURESPROC                           glBindTextures;
    PFNGLBINDSAMPLERSPROC                           glBindSamplers;
    PFNGLBINDIMAGETEXTURESPROC                      glBindImageTextures;
    PFNGLBINDVERTEXBUFFERSPROC                      glBindVertexBuffers;
    // ARB_query_buffer_object (no entry points)
    // ARB_texture_mirror_clamp_to_edge (no entry points)
    // ARB_texture_stencil8 (no entry points)
    // ARB_vertex_type_10f_11f_11f_rev (no entry points)

    // GL_ARB_shading_language_include
    PFNGLNAMEDSTRINGARBPROC                         glNamedStringARB;
    PFNGLDELETENAMEDSTRINGARBPROC                   glDeleteNamedStringARB;
    PFNGLCOMPILESHADERINCLUDEARBPROC                glCompileShaderIncludeARB;
    PFNGLISNAMEDSTRINGARBPROC                       glIsNamedStringARB;
    PFNGLGETNAMEDSTRINGARBPROC                      glGetNamedStringARB;
    PFNGLGETNAMEDSTRINGIVARBPROC                    glGetNamedStringivARB;

    // ARB_cl_event
    PFNGLCREATESYNCFROMCLEVENTARBPROC               glCreateSyncFromCLeventARB;

    // ARB_debug_output
    PFNGLDEBUGMESSAGECONTROLARBPROC                 glDebugMessageControlARB;
    PFNGLDEBUGMESSAGEINSERTARBPROC                  glDebugMessageInsertARB;
    PFNGLDEBUGMESSAGECALLBACKARBPROC                glDebugMessageCallbackARB;
    PFNGLGETDEBUGMESSAGELOGARBPROC                  glGetDebugMessageLogARB;

    // ARB_robustness
    PFNGLGETGRAPHICSRESETSTATUSARBPROC              glGetGraphicsResetStatusARB;
    //PFNGLGETNMAPDVARBPROC                           glGetnMapdvARB;
    //PFNGLGETNMAPFVARBPROC                           glGetnMapfvARB;
    //PFNGLGETNMAPIVARBPROC                           glGetnMapivARB;
    //PFNGLGETNPIXELMAPFVARBPROC                      glGetnPixelMapfvARB;
    //PFNGLGETNPIXELMAPUIVARBPROC                     glGetnPixelMapuivARB;
    //PFNGLGETNPIXELMAPUSVARBPROC                     glGetnPixelMapusvARB;
    //PFNGLGETNPOLYGONSTIPPLEARBPROC                  glGetnPolygonStippleARB;
    //PFNGLGETNCOLORTABLEARBPROC                      glGetnColorTableARB;
    //PFNGLGETNCONVOLUTIONFILTERARBPROC               glGetnConvolutionFilterARB;
    //PFNGLGETNSEPARABLEFILTERARBPROC                 glGetnSeparableFilterARB;
    //PFNGLGETNHISTOGRAMARBPROC                       glGetnHistogramARB;
    //PFNGLGETNMINMAXARBPROC                          glGetnMinmaxARB;
    PFNGLGETNTEXIMAGEARBPROC                        glGetnTexImageARB;
    PFNGLREADNPIXELSARBPROC                         glReadnPixelsARB;
    PFNGLGETNCOMPRESSEDTEXIMAGEARBPROC              glGetnCompressedTexImageARB;
    PFNGLGETNUNIFORMFVARBPROC                       glGetnUniformfvARB;
    PFNGLGETNUNIFORMIVARBPROC                       glGetnUniformivARB;
    PFNGLGETNUNIFORMUIVARBPROC                      glGetnUniformuivARB;
    PFNGLGETNUNIFORMDVARBPROC                       glGetnUniformdvARB;


    // EXT_shader_image_load_store
    PFNGLBINDIMAGETEXTUREEXTPROC                    glBindImageTextureEXT;
    PFNGLMEMORYBARRIEREXTPROC                       glMemoryBarrierEXT;

    // EXT_direct_state_access
    // use GL_EXT_draw_buffers2
    PFNGLENABLEINDEXEDEXTPROC                       glEnableIndexedEXT;
    PFNGLDISABLEINDEXEDEXTPROC                      glDisableIndexedEXT;

    // buffer handling ////////////////////////////////////////////////////////////////////////////
    PFNGLNAMEDBUFFERDATAEXTPROC                     glNamedBufferDataEXT;
    PFNGLNAMEDBUFFERSUBDATAEXTPROC                  glNamedBufferSubDataEXT;
    PFNGLMAPNAMEDBUFFEREXTPROC                      glMapNamedBufferEXT;
    PFNGLUNMAPNAMEDBUFFEREXTPROC                    glUnmapNamedBufferEXT;
    PFNGLGETNAMEDBUFFERPARAMETERIVEXTPROC           glGetNamedBufferParameterivEXT;
    PFNGLGETNAMEDBUFFERPOINTERVEXTPROC              glGetNamedBufferPointervEXT;
    PFNGLGETNAMEDBUFFERSUBDATAEXTPROC               glGetNamedBufferSubDataEXT;

    PFNGLMAPNAMEDBUFFERRANGEEXTPROC                 glMapNamedBufferRangeEXT;
    PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEEXTPROC         glFlushMappedNamedBufferRangeEXT;

    PFNGLNAMEDCOPYBUFFERSUBDATAEXTPROC              glNamedCopyBufferSubDataEXT;

    PFNGLVERTEXARRAYVERTEXATTRIBOFFSETEXTPROC       glVertexArrayVertexAttribOffsetEXT;
    PFNGLVERTEXARRAYVERTEXATTRIBIOFFSETEXTPROC      glVertexArrayVertexAttribIOffsetEXT;
    PFNGLENABLEVERTEXARRAYATTRIBEXTPROC             glEnableVertexArrayAttribEXT;
    PFNGLDISABLEVERTEXARRAYATTRIBEXTPROC            glDisableVertexArrayAttribEXT;

    // shader handling ////////////////////////////////////////////////////////////////////////////
    PFNGLPROGRAMUNIFORM1FEXTPROC                    glProgramUniform1fEXT;
    PFNGLPROGRAMUNIFORM2FEXTPROC                    glProgramUniform2fEXT;
    PFNGLPROGRAMUNIFORM3FEXTPROC                    glProgramUniform3fEXT;
    PFNGLPROGRAMUNIFORM4FEXTPROC                    glProgramUniform4fEXT;
    PFNGLPROGRAMUNIFORM1IEXTPROC                    glProgramUniform1iEXT;
    PFNGLPROGRAMUNIFORM2IEXTPROC                    glProgramUniform2iEXT;
    PFNGLPROGRAMUNIFORM3IEXTPROC                    glProgramUniform3iEXT;
    PFNGLPROGRAMUNIFORM4IEXTPROC                    glProgramUniform4iEXT;
    PFNGLPROGRAMUNIFORM1FVEXTPROC                   glProgramUniform1fvEXT;
    PFNGLPROGRAMUNIFORM2FVEXTPROC                   glProgramUniform2fvEXT;
    PFNGLPROGRAMUNIFORM3FVEXTPROC                   glProgramUniform3fvEXT;
    PFNGLPROGRAMUNIFORM4FVEXTPROC                   glProgramUniform4fvEXT;
    PFNGLPROGRAMUNIFORM1IVEXTPROC                   glProgramUniform1ivEXT;
    PFNGLPROGRAMUNIFORM2IVEXTPROC                   glProgramUniform2ivEXT;
    PFNGLPROGRAMUNIFORM3IVEXTPROC                   glProgramUniform3ivEXT;
    PFNGLPROGRAMUNIFORM4IVEXTPROC                   glProgramUniform4ivEXT;
    PFNGLPROGRAMUNIFORMMATRIX2FVEXTPROC             glProgramUniformMatrix2fvEXT;
    PFNGLPROGRAMUNIFORMMATRIX3FVEXTPROC             glProgramUniformMatrix3fvEXT;
    PFNGLPROGRAMUNIFORMMATRIX4FVEXTPROC             glProgramUniformMatrix4fvEXT;
    PFNGLPROGRAMUNIFORMMATRIX2X3FVEXTPROC           glProgramUniformMatrix2x3fvEXT;
    PFNGLPROGRAMUNIFORMMATRIX3X2FVEXTPROC           glProgramUniformMatrix3x2fvEXT;
    PFNGLPROGRAMUNIFORMMATRIX2X4FVEXTPROC           glProgramUniformMatrix2x4fvEXT;
    PFNGLPROGRAMUNIFORMMATRIX4X2FVEXTPROC           glProgramUniformMatrix4x2fvEXT;
    PFNGLPROGRAMUNIFORMMATRIX3X4FVEXTPROC           glProgramUniformMatrix3x4fvEXT;
    PFNGLPROGRAMUNIFORMMATRIX4X3FVEXTPROC           glProgramUniformMatrix4x3fvEXT;
    PFNGLPROGRAMUNIFORM1UIEXTPROC                   glProgramUniform1uiEXT;
    PFNGLPROGRAMUNIFORM2UIEXTPROC                   glProgramUniform2uiEXT;
    PFNGLPROGRAMUNIFORM3UIEXTPROC                   glProgramUniform3uiEXT;
    PFNGLPROGRAMUNIFORM4UIEXTPROC                   glProgramUniform4uiEXT;
    PFNGLPROGRAMUNIFORM1UIVEXTPROC                  glProgramUniform1uivEXT;
    PFNGLPROGRAMUNIFORM2UIVEXTPROC                  glProgramUniform2uivEXT;
    PFNGLPROGRAMUNIFORM3UIVEXTPROC                  glProgramUniform3uivEXT;
    PFNGLPROGRAMUNIFORM4UIVEXTPROC                  glProgramUniform4uivEXT;

    // texture handling ///////////////////////////////////////////////////////////////////////////
    PFNGLTEXTUREPARAMETERFEXTPROC                   glTextureParameterfEXT;
    PFNGLTEXTUREPARAMETERFVEXTPROC                  glTextureParameterfvEXT;
    PFNGLTEXTUREPARAMETERIEXTPROC                   glTextureParameteriEXT;
    PFNGLTEXTUREPARAMETERIVEXTPROC                  glTextureParameterivEXT;
    PFNGLTEXTUREPARAMETERIIVEXTPROC                 glTextureParameterIivEXT;
    PFNGLTEXTUREPARAMETERIUIVEXTPROC                glTextureParameterIuivEXT;

    PFNGLTEXTUREIMAGE1DEXTPROC                      glTextureImage1DEXT;
    PFNGLTEXTUREIMAGE2DEXTPROC                      glTextureImage2DEXT;
    PFNGLTEXTUREIMAGE3DEXTPROC                      glTextureImage3DEXT;
    PFNGLTEXTURESUBIMAGE1DEXTPROC                   glTextureSubImage1DEXT;
    PFNGLTEXTURESUBIMAGE2DEXTPROC                   glTextureSubImage2DEXT;
    PFNGLTEXTURESUBIMAGE3DEXTPROC                   glTextureSubImage3DEXT;
    PFNGLCOMPRESSEDTEXTURESUBIMAGE1DEXTPROC         glCompressedTextureSubImage1DEXT;
    PFNGLCOMPRESSEDTEXTURESUBIMAGE2DEXTPROC         glCompressedTextureSubImage2DEXT;
    PFNGLCOMPRESSEDTEXTURESUBIMAGE3DEXTPROC         glCompressedTextureSubImage3DEXT;

    PFNGLGETTEXTUREIMAGEEXTPROC                     glGetTextureImageEXT;
    PFNGLGETTEXTUREPARAMETERFVEXTPROC               glGetTextureParameterfvEXT;
    PFNGLGETTEXTUREPARAMETERIVEXTPROC               glGetTextureParameterivEXT;

    PFNGLTEXTUREBUFFEREXTPROC                       glTextureBufferEXT;
    PFNGLMULTITEXBUFFEREXTPROC                      glMultiTexBufferEXT;
    PFNGLBINDMULTITEXTUREEXTPROC                    glBindMultiTextureEXT;

    // frame buffer handling //////////////////////////////////////////////////////////////////////
    PFNGLNAMEDRENDERBUFFERSTORAGEEXTPROC                    glNamedRenderbufferStorageEXT;
    PFNGLGETNAMEDRENDERBUFFERPARAMETERIVEXTPROC             glGetNamedRenderbufferParameterivEXT;
    PFNGLCHECKNAMEDFRAMEBUFFERSTATUSEXTPROC                 glCheckNamedFramebufferStatusEXT;
    PFNGLNAMEDFRAMEBUFFERTEXTURE1DEXTPROC                   glNamedFramebufferTexture1DEXT;
    PFNGLNAMEDFRAMEBUFFERTEXTURE2DEXTPROC                   glNamedFramebufferTexture2DEXT;
    PFNGLNAMEDFRAMEBUFFERTEXTURE3DEXTPROC                   glNamedFramebufferTexture3DEXT;
    PFNGLNAMEDFRAMEBUFFERRENDERBUFFEREXTPROC                glNamedFramebufferRenderbufferEXT;
    PFNGLGETNAMEDFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC    glGetNamedFramebufferAttachmentParameterivEXT;
    PFNGLGENERATETEXTUREMIPMAPEXTPROC                       glGenerateTextureMipmapEXT;
    PFNGLFRAMEBUFFERDRAWBUFFEREXTPROC                       glFramebufferDrawBufferEXT;
    PFNGLFRAMEBUFFERDRAWBUFFERSEXTPROC                      glFramebufferDrawBuffersEXT;
    PFNGLFRAMEBUFFERREADBUFFEREXTPROC                       glFramebufferReadBufferEXT;
    PFNGLGETFRAMEBUFFERPARAMETERIVEXTPROC                   glGetFramebufferParameterivEXT;
    PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEEXTPROC         glNamedRenderbufferStorageMultisampleEXT;
    PFNGLNAMEDFRAMEBUFFERTEXTUREEXTPROC                     glNamedFramebufferTextureEXT;
    PFNGLNAMEDFRAMEBUFFERTEXTURELAYEREXTPROC                glNamedFramebufferTextureLayerEXT;

    // GL_NV_bindless_texture
    PFNGLGETTEXTUREHANDLENVPROC                     glGetTextureHandleNV;
    PFNGLGETTEXTURESAMPLERHANDLENVPROC              glGetTextureSamplerHandleNV;
    PFNGLMAKETEXTUREHANDLERESIDENTNVPROC            glMakeTextureHandleResidentNV;
    PFNGLMAKETEXTUREHANDLENONRESIDENTNVPROC         glMakeTextureHandleNonResidentNV;
    PFNGLGETIMAGEHANDLENVPROC                       glGetImageHandleNV;
    PFNGLMAKEIMAGEHANDLERESIDENTNVPROC              glMakeImageHandleResidentNV;
    PFNGLMAKEIMAGEHANDLENONRESIDENTNVPROC           glMakeImageHandleNonResidentNV;
    PFNGLUNIFORMHANDLEUI64NVPROC                    glUniformHandleui64NV;
    PFNGLUNIFORMHANDLEUI64VNVPROC                   glUniformHandleui64vNV;
    PFNGLPROGRAMUNIFORMHANDLEUI64NVPROC             glProgramUniformHandleui64NV;
    PFNGLPROGRAMUNIFORMHANDLEUI64VNVPROC            glProgramUniformHandleui64vNV;
    PFNGLISTEXTUREHANDLERESIDENTNVPROC              glIsTextureHandleResidentNV;
    PFNGLISIMAGEHANDLERESIDENTNVPROC                glIsImageHandleResidentNV;

    // GL_ARB_bindless_texture
    PFNGLGETTEXTUREHANDLEARBPROC                    glGetTextureHandleARB;
    PFNGLGETTEXTURESAMPLERHANDLEARBPROC             glGetTextureSamplerHandleARB;
    PFNGLMAKETEXTUREHANDLERESIDENTARBPROC           glMakeTextureHandleResidentARB;
    PFNGLMAKETEXTUREHANDLENONRESIDENTARBPROC        glMakeTextureHandleNonResidentARB;
    PFNGLGETIMAGEHANDLEARBPROC                      glGetImageHandleARB;
    PFNGLMAKEIMAGEHANDLERESIDENTARBPROC             glMakeImageHandleResidentARB;
    PFNGLMAKEIMAGEHANDLENONRESIDENTARBPROC          glMakeImageHandleNonResidentARB;
    PFNGLUNIFORMHANDLEUI64ARBPROC                   glUniformHandleui64ARB;
    PFNGLUNIFORMHANDLEUI64VARBPROC                  glUniformHandleui64vARB;
    PFNGLPROGRAMUNIFORMHANDLEUI64ARBPROC            glProgramUniformHandleui64ARB;
    PFNGLPROGRAMUNIFORMHANDLEUI64VARBPROC           glProgramUniformHandleui64vARB;
    PFNGLISTEXTUREHANDLERESIDENTARBPROC             glIsTextureHandleResidentARB;
    PFNGLISIMAGEHANDLERESIDENTARBPROC               glIsImageHandleResidentARB;
    PFNGLVERTEXATTRIBL1UI64ARBPROC                  glVertexAttribL1ui64ARB;
    PFNGLVERTEXATTRIBL1UI64VARBPROC                 glVertexAttribL1ui64vARB;
    PFNGLGETVERTEXATTRIBLUI64VARBPROC               glGetVertexAttribLui64vARB;

    // GL_NV_shader_buffer_load
    PFNGLMAKEBUFFERRESIDENTNVPROC                   glMakeBufferResidentNV;
    PFNGLMAKEBUFFERNONRESIDENTNVPROC                glMakeBufferNonResidentNV;
    PFNGLISBUFFERRESIDENTNVPROC                     glIsBufferResidentNV;
    PFNGLMAKENAMEDBUFFERRESIDENTNVPROC              glMakeNamedBufferResidentNV;
    PFNGLMAKENAMEDBUFFERNONRESIDENTNVPROC           glMakeNamedBufferNonResidentNV;
    PFNGLISNAMEDBUFFERRESIDENTNVPROC                glIsNamedBufferResidentNV;
    PFNGLGETBUFFERPARAMETERUI64VNVPROC              glGetBufferParameterui64vNV;
    PFNGLGETNAMEDBUFFERPARAMETERUI64VNVPROC         glGetNamedBufferParameterui64vNV;
    PFNGLGETINTEGERUI64VNVPROC                      glGetIntegerui64vNV;
    PFNGLUNIFORMUI64NVPROC                          glUniformui64NV;
    PFNGLUNIFORMUI64VNVPROC                         glUniformui64vNV;
    PFNGLGETUNIFORMUI64VNVPROC                      glGetUniformui64vNV;
    PFNGLPROGRAMUNIFORMUI64NVPROC                   glProgramUniformui64NV;
    PFNGLPROGRAMUNIFORMUI64VNVPROC                  glProgramUniformui64vNV;

    // ARB_sparse_texture
    PFNGLTEXPAGECOMMITMENTARBPROC                   glTexPageCommitmentARB;
    PFNGLTEXTUREPAGECOMMITMENTEXTPROC               glTexturePageCommitmentEXT;

    // ARB_compute_variable_group_size
    PFNGLDISPATCHCOMPUTEGROUPSIZEARBPROC            glDispatchComputeGroupSizeARB;

    // EXT_raster_multisample
    PFNGLRASTERSAMPLESEXTPROC                       glRasterSamplesEXT;

    // NV_framebuffer_mixed_samples
    PFNGLCOVERAGEMODULATIONTABLENVPROC              glCoverageModulationTableNV;
    PFNGLGETCOVERAGEMODULATIONTABLENVPROC           glGetCoverageModulationTableNV;
    PFNGLCOVERAGEMODULATIONNVPROC                   glCoverageModulationNV;

    // NV_fragment_coverage_to_color
    PFNGLFRAGMENTCOVERAGECOLORNVPROC                glFragmentCoverageColorNV;

    // NV_sample_locations
    PFNGLFRAMEBUFFERSAMPLELOCATIONSFVNVPROC         glFramebufferSampleLocationsfvNV;
    PFNGLNAMEDFRAMEBUFFERSAMPLELOCATIONSFVNVPROC    glNamedFramebufferSampleLocationsfvNV;
    PFNGLRESOLVEDEPTHVALUESNVPROC                   glResolveDepthValuesNV;

    // NV_conservative_raster
    PFNGLSUBPIXELPRECISIONBIASNVPROC                glSubpixelPrecisionBiasNV;

}; // class gl_core

} // namespace opengl
} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_OPENGL_DETAIL_GL_CORE_H_INCLUDED
