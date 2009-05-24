
#include "window_context.h"

namespace scm {
namespace gl {

window_context::window_context()
{
}

window_context::~window_context()
{
}

bool
window_context::operator==(const window_context& rhs) const
{
    bool tmp_ret = true;

    tmp_ret = tmp_ret && (_context_format == rhs._context_format);
    tmp_ret = tmp_ret && (_context_handle == rhs._context_handle);

    return (tmp_ret);
}

bool
window_context::operator!=(const window_context& rhs) const
{
    return (!(*this == rhs));
}

} // namespace gl
} // namespace scm
