
#include "file.h"

namespace scm {
namespace io {


file::file()
  : detail::file_impl()
{
}

file::file(const file& rhs)
  : detail::file_impl(rhs)
{
}

file::~file()
{
}

file&
file::operator=(const file& rhs)
{
    file tmp(rhs);

    swap(tmp);

    return (*this);
}

} // namespace io
} // namespace scm
