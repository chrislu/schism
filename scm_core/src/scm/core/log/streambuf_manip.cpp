
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "streambuf_manip.h"

#include <scm/core/utilities/static_global.h>

namespace  {

struct ios_base_log_streambuf_index
{
    ios_base_log_streambuf_index() {
        _index = std::ios_base::xalloc();
    }
    int _index;
};

SCM_STATIC_GLOBAL(ios_base_log_streambuf_index, global_ios_base_log_streambuf_index)

} // namespace 

namespace scm {
namespace string {
namespace detail {

int log_streambuf_index()
{
    return (global_ios_base_log_streambuf_index()._index);
}

} // namespace detail
} // namespace string
} // namespace scm
