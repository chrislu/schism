
#include "context_base.h"

namespace scm {
namespace gl {

context_base::context_base()
  : _context_format(context_format::null_format())
{
}

context_base::~context_base()
{
}

const context_format&
context_base::format() const
{
    return (_context_format);
}

const context_base::handle&
context_base::context_handle() const
{
    return (_context_handle);
}
const context_base::handle&
context_base::device_handle() const
{
    return (_device_handle);
}

bool
context_base::empty() const
{
    return (_context_handle.get() == 0);
}

} // namespace gl
} // namespace scm
