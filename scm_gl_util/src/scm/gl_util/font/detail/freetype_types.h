
#ifndef SCM_GL_UTIL_DETAIL_FREETYPE_TYPES_H_INCLUDED
#define SCM_GL_UTIL_DETAIL_FREETYPE_TYPES_H_INCLUDED

#include <string>
#include <vector>

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H
#include FT_OUTLINE_H
#include FT_STROKER_H

namespace scm {
namespace gl {
namespace detail {

struct glyph_span{
    glyph_span(const short x,
               const short y,
               const unsigned short len,
               const unsigned char coverage,
               const bool core)
      : _x(x)
      , _y(y)
      , _length(len)
      , _coverage(coverage)
      , _core(core) {}

    short           _x;
    short           _y;
    unsigned short _length;
    unsigned char  _coverage;
    bool           _core;
};

typedef std::vector<glyph_span> span_vector;

template <bool core>
void span_func(const int            y,
                const int            count,
                const FT_Span* const spans,
                void*const           user) {
    span_vector*const spans = static_cast<span_vector*>(user);

    for (int i = 0; i < count; ++i) {
        spans->push_back(glyph_span(spans[i].x,
                            static_cast<short>(y),
                            spans[i].len,
                            spans[i].coverage,
                            core));
    }
}

class ft_library
{
public:
    ft_library();
    /*virtual*/ ~ft_library();

    bool                open();
    const FT_Library    get_lib() const { return (_lib); }

    template <bool core>
    bool outline_render(FT_Outline* const outline, span_vector& spans) const {
        FT_Raster_Params params;

        params.target        = NULL;
        params.source        = NULL;
        params.black_spans   = NULL;
        params.bit_test      = NULL;
        params.bit_set       = NULL;
        params.clip_box.xMin = 0;
        params.clip_box.yMin = 0;
        params.clip_box.xMax = 0;
        params.clip_box.yMax = 0;

        params.flags      = FT_RASTER_FLAG_AA | FT_RASTER_FLAG_DIRECT;
        params.gray_spans = span_func<core>;
        params.user       = &spans;

        if (FT_Outline_Render(m_library, outline, &params)) {
            return (false);
        }
        return (true);
    }

protected:
    FT_Library          _lib;
}; // class ft_library

class ft_face
{
public:
    ft_face();
    /*virtual*/ ~ft_face();

    bool                open_face(const ft_library&  /*lib*/,
                                  const std::string& /*file*/);

    const FT_Face       get_face() const { return (_face); }

protected:
    FT_Face             _face;
}; // class ft_face

class ft_stroker
{
public:
    ft_stroker(const ft_library&  /*lib*/,
               unsigned           /*border_size*/);
    /*virtual*/ ~ft_stroker();

    const FT_Stroker    get_stroker() const { return (_stroker); }

protected:
    FT_Stroker      _stroker;
}; // class ft_stroker

} // namespace detail
} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_DETAIL_FREETYPE_TYPES_H_INCLUDED
