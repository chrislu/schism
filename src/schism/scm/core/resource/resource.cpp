
#include "resource.h"

namespace scm {
namespace res {

resource_base::resource_base()
{
}

resource_base::~resource_base()
{
}

bool resource_base::operator==(const resource_base& rhs) const
{
    return (hash_value() == rhs.hash_value());
}

} // namespace res
} // namespace scm
