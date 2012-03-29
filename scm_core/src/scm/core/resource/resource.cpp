
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

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
