
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "volume_reader.h"

namespace scm {
namespace gl {

volume_reader::volume_reader(const std::string& file_path,
                                   bool         file_unbuffered)
  : _dimensions(math::vec3ui(0u))
  , _format(FORMAT_NULL)
  , _file_path(file_path)
{
}

volume_reader::~volume_reader()
{
}

const data_format
volume_reader::format() const
{
    return _format;
}

const math::vec3ui&
volume_reader::dimensions() const
{
    return _dimensions;
}

volume_reader::operator bool() const
{
    return _file.get() != 0;
}

bool
volume_reader::operator! () const
{
    return _file.get() == 0;
}

} // namespace gl
} // namespace scm
