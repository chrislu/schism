
#ifndef SCM_GL_CORE_OPENGL_DETAIL_GL3_CORE_H_INCLUDED
#define SCM_GL_CORE_OPENGL_DETAIL_GL3_CORE_H_INCLUDED

#include <set>
#include <string>

#include <scm/gl_core/opengl/gl/gl3.h>
#include <scm/gl_core/opengl/gl/glext.h>

// temporary
//#include <scm/core/platform/platform.h>
//#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
namespace opengl {

class gl3_core
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

        std::string     _vendor;
        std::string     _renderer;

        profile_type    _profile;
        std::string     _profile_string;

        context_info() : _version_major(0), _version_minor(0), _version_release(0), _profile(profile_unknown) {}
    };

    // glext.h missing types
    typedef GLvoid * (APIENTRY *PFNGLMAPNAMEDBUFFERRANGEEXTPROC) (GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access);
    typedef void (APIENTRY *PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEEXTPROC) (GLuint buffer, GLintptr offset, GLsizeiptr length);

private:
    typedef std::set<std::string>   string_set;

public:
    gl3_core();

    bool                    initialize();
    const context_info&     context_information() const;
    bool                    is_initialized() const;
    bool                    is_supported(const std::string& ext) const;

private:
    void            init_entry_points();

private:
    string_set      _extensions;
    bool            _initialized;
    context_info    _context_info;

    friend std::ostream& operator<<(std::ostream& out_stream, const gl3_core& c);

public:
    // version 1.0 ////////////////////////////////////////////////////////////////////////////////
    bool    version_1_0_available;
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
    bool    version_1_1_available;
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
    bool    version_1_2_available;
    PFNGLBLENDCOLORPROC                             glBlendColor;
    PFNGLBLENDEQUATIONPROC                          glBlendEquation;
    PFNGLDRAWRANGEELEMENTSPROC                      glDrawRangeElements;
    PFNGLTEXIMAGE3DPROC                             glTexImage3D;
    PFNGLTEXSUBIMAGE3DPROC                          glTexSubImage3D;
    PFNGLCOPYTEXSUBIMAGE3DPROC                      glCopyTexSubImage3D;
                                            
    // version 1.3 ////////////////////////////////////////////////////////////////////////////////
    bool    version_1_3_available;
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
    bool    version_1_4_available;
    PFNGLBLENDFUNCSEPARATEPROC                      glBlendFuncSeparate;
    PFNGLMULTIDRAWARRAYSPROC                        glMultiDrawArrays;
    PFNGLMULTIDRAWELEMENTSPROC                      glMultiDrawElements;
    PFNGLPOINTPARAMETERFPROC                        glPointParameterf;
    PFNGLPOINTPARAMETERFVPROC                       glPointParameterfv;
    PFNGLPOINTPARAMETERIPROC                        glPointParameteri;
    PFNGLPOINTPARAMETERIVPROC                       glPointParameteriv;
                                            
    // version 1.5 ////////////////////////////////////////////////////////////////////////////////
    bool    version_1_5_available;
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
    bool    version_2_0_available;
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
    bool    version_2_1_available;
    PFNGLUNIFORMMATRIX2X3FVPROC                     glUniformMatrix2x3fv;
    PFNGLUNIFORMMATRIX3X2FVPROC                     glUniformMatrix3x2fv;
    PFNGLUNIFORMMATRIX2X4FVPROC                     glUniformMatrix2x4fv;
    PFNGLUNIFORMMATRIX4X2FVPROC                     glUniformMatrix4x2fv;
    PFNGLUNIFORMMATRIX3X4FVPROC                     glUniformMatrix3x4fv;
    PFNGLUNIFORMMATRIX4X3FVPROC                     glUniformMatrix4x3fv;

    // version 3.0 ////////////////////////////////////////////////////////////////////////////////
    bool    version_3_0_available;
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
    bool    version_3_1_available;
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
    bool    version_3_2_available;
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
    bool    version_3_3_available;
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
    PFNGLGETSAMPLERPARAMETERIFVPROC                 glGetSamplerParameterIfv;
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
    bool    version_4_0_available;
    // work in progress

    // EXT_direct_state_access
    // buffer handling ////////////////////////////////////////////////////////////////////////////
    bool    extension_EXT_direct_state_access_available;
    PFNGLNAMEDBUFFERDATAEXTPROC                     glNamedBufferDataEXT;
    PFNGLNAMEDBUFFERSUBDATAEXTPROC                  glNamedBufferSubDataEXT;
    PFNGLMAPNAMEDBUFFEREXTPROC                      glMapNamedBufferEXT;
    PFNGLUNMAPNAMEDBUFFEREXTPROC                    glUnmapNamedBufferEXT;
    PFNGLGETNAMEDBUFFERPARAMETERIVEXTPROC           glGetNamedBufferParameterivEXT;
    PFNGLGETNAMEDBUFFERPOINTERVEXTPROC              glGetNamedBufferPointervEXT;
    PFNGLGETNAMEDBUFFERSUBDATAEXTPROC               glGetNamedBufferSubDataEXT;

    PFNGLMAPNAMEDBUFFERRANGEEXTPROC                 glMapNamedBufferRangeEXT;
    PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEEXTPROC         glFlushMappedNamedBufferRangeEXT;

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

}; // class gl3_core

} // namespace opengl
} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_OPENGL_DETAIL_GL3_CORE_H_INCLUDED
