
#include "context.h"

namespace scm {
namespace gl {

context::context()
  : _context_format(context_format::null_format())
{
}

context::~context()
{
}

const context_format&
context::format() const
{
    return (_context_format);
}

const context::handle&
context::context_handle() const
{
    return (_context_handle);
}

bool
context::operator==(const context& rhs) const
{
    bool tmp_ret = true;

    tmp_ret = tmp_ret && (_context_format == rhs._context_format);
    tmp_ret = tmp_ret && (_context_handle == rhs._context_handle);

    return (tmp_ret);
}

bool
context::operator!=(const context& rhs) const
{
    return (!(*this == rhs));
}

bool
context::empty() const
{
    return (_context_handle.get() == 0);
}

} // namespace gl
} // namespace scm
