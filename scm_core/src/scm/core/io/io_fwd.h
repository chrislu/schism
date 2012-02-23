
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_IO_FWD_H_INCLUDED
#define SCM_CORE_IO_FWD_H_INCLUDED

#include <scm/core/memory.h>

namespace scm {
namespace io {

class file;

typedef scm::int64      size_type;
typedef size_type       offset_type;

typedef shared_ptr<file>        file_ptr;
typedef shared_ptr<const file>  file_const_ptr;
typedef weak_ptr<file>          file_weak_ptr;
typedef weak_ptr<const file>    file_weak_const_ptr;

} // namespace io
} // namespace scm

#endif // SCM_CORE_IO_FWD_H_INCLUDED
