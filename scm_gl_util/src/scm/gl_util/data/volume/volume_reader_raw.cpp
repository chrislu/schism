
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "volume_reader_raw.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>

#include <scm/core/io/file.h>

#include <scm/gl_core/log.h>

namespace {

bool parse_raw_file_name(const std::string& filename,
                         unsigned& file_offset,
                         unsigned& width,
                         unsigned& height,
                         unsigned& depth,
                         unsigned& num_components,
                         unsigned& bit_per_voxel)
{
    using namespace scm::gl;

    namespace qi = boost::spirit::qi;
    namespace ph = boost::phoenix;

    std::string::const_iterator b = filename.begin();
    std::string::const_iterator e = filename.end();

    file_offset = 0;

    qi::rule<std::string::const_iterator> raw_info =
         -(qi::lit("_o") >> qi::uint_[ph::ref(file_offset) = qi::_1])
        >> qi::lit("_w") >> qi::uint_[ph::ref(width) = qi::_1]
        >> qi::lit("_h") >> qi::uint_[ph::ref(height) = qi::_1]
        >> qi::lit("_d") >> qi::uint_[ph::ref(depth) = qi::_1]
        >> qi::lit("_c") >> qi::uint_[ph::ref(num_components) = qi::_1]
        >> qi::lit("_b") >> qi::uint_[ph::ref(bit_per_voxel) = qi::_1];

    qi::rule<std::string::const_iterator> raw_file_name_format =
        +((raw_info >> qi::lit(".raw")) | (*qi::char_('_') >> (qi::char_ - qi::char_('_'))));

    if (   !qi::phrase_parse(b, e, raw_file_name_format, boost::spirit::ascii::space)
        || b != e) {
        glerr() << scm::log::error
                << "volume_reader_raw::parse_raw_file_name(): "
                << "unable to parse raw file name format, malformed file name string ('"
                << filename << "')" << scm::log::end;
        return false;
    }

    return true;
}

} // namespace


namespace scm {
namespace gl {

volume_reader_raw::volume_reader_raw(const std::string& file_path,
                                           bool         file_unbuffered)
  : volume_reader_blocked(file_path, file_unbuffered)
{
    using namespace boost::filesystem;

    path            fpath(file_path);
    std::string     fname = fpath.filename().string();
    std::string     fext  = fpath.extension().string();

    unsigned doffset  = 0;
    unsigned dnumchan = 0;
    unsigned dbpp     = 0;

    if (!parse_raw_file_name(fpath.string(),
                             doffset,
                             _dimensions.x,
                             _dimensions.y,
                             _dimensions.z,
                             dnumchan,
                             dbpp))
    {
        return;
    }

    _data_start_offset = doffset;

    switch (dnumchan) {
    case 1:
        switch (dbpp) {
        case 8:  _format = FORMAT_R_8;   break;
        case 16: _format = FORMAT_R_16;  break;
        case 32: _format = FORMAT_R_32F; break;
        }
        break;
    case 2:
        switch (dbpp) {
        case 8:  _format = FORMAT_RG_8;   break;
        case 16: _format = FORMAT_RG_16;  break;
        case 32: _format = FORMAT_RG_32F; break;
        }
        break;
    case 3:
        switch (dbpp) {
        case 8:  _format = FORMAT_RGB_8;   break;
        case 16: _format = FORMAT_RGB_16;  break;
        case 32: _format = FORMAT_RGB_32F; break;
        }
        break;
    case 4:
        switch (dbpp) {
        case 8:  _format = FORMAT_RGBA_8;   break;
        case 16: _format = FORMAT_RGBA_16;  break;
        case 32: _format = FORMAT_RGBA_32F; break;
        }
        break;
    }

    if (_format == FORMAT_NULL) {
        glerr() << scm::log::error
                << "volume_reader_raw::volume_reader_raw(): "
                << "unable match data format (channels: " << dnumchan << ", bpp: " << dbpp << ")." << scm::log::end;
        return;
    }

    _file = make_shared<io::file>();
    
    if (!_file->open(fpath.string(), std::ios_base::in, file_unbuffered)) {
        _file.reset();
        glerr() << scm::log::error
                << "volume_reader_raw::volume_reader_raw(): "
                << "error opening volume file (" << fpath.string() << ")." << scm::log::end;
        return;
    }

    // check if filesize checks out with given dimensions
    size_t expect_fs =   static_cast<size_t>(_dimensions.x)
                         * static_cast<size_t>(_dimensions.y)
                         * static_cast<size_t>(_dimensions.z)
                         * size_of_format(_format)
                         + _data_start_offset;
    if (_file->size() != expect_fs) {
        _file.reset();

        glerr() << scm::log::error
                << "volume_reader_raw::volume_reader_raw(): "
                << "file size does not match data dimensions and data format"
                << " (file_size: " << _file->size()
                << ", expected size: " << expect_fs << ")." << scm::log::end;
        return;
    }

    size_t slice_size = static_cast<size_t>(_dimensions.x) * _dimensions.y * size_of_format(_format);
    _slice_buffer.reset(new uint8[slice_size]);
}

volume_reader_raw::volume_reader_raw(
        const std::string&   file_path,
        const math::vec3ui&  volume_dimensions,
        scm::gl::data_format voxel_fmt,
        bool                 file_unbuffered)
  : volume_reader_blocked(file_path, file_unbuffered)
{
    using namespace boost::filesystem;

    path            fpath(file_path);
    std::string     fname = fpath.filename().string();
    std::string     fext  = fpath.extension().string();

    _format = voxel_fmt;
    if (_format == FORMAT_NULL) {
        glerr() << scm::log::error
                << "volume_reader_raw::volume_reader_raw(): invalid voxel format)." << scm::log::end;
        return;
    }

    _file = make_shared<io::file>();
    
    if (!_file->open(fpath.string(), std::ios_base::in, file_unbuffered)) {
        _file.reset();
        glerr() << scm::log::error
                << "volume_reader_raw::volume_reader_raw(): "
                << "error opening volume file (" << fpath.string() << ")." << scm::log::end;
        return;
    }

    _dimensions = volume_dimensions;
    // check if filesize checks out with given dimensions
    size_t expect_fs =     static_cast<size_t>(_dimensions.x)
                         * static_cast<size_t>(_dimensions.y)
                         * static_cast<size_t>(_dimensions.z)
                         * size_of_format(_format)
                         + _data_start_offset;
    if (_file->size() != expect_fs) {
        _file.reset();

        glerr() << scm::log::error
                << "volume_reader_raw::volume_reader_raw(): "
                << "file size does not match data dimensions and data format"
                << " (file_size: " << _file->size()
                << ", expected size: " << expect_fs << ")." << scm::log::end;
        return;
    }

    size_t slice_size = static_cast<size_t>(_dimensions.x) * _dimensions.y * size_of_format(_format);
    _slice_buffer.reset(new uint8[slice_size]);
}

volume_reader_raw::~volume_reader_raw()
{
    _slice_buffer.reset();
    _file->close();
    _file.reset();
}

} // namespace gl
} // namespace scm
