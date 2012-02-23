
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_INPUT_DEVICES_FWD_H_INCLUDED
#define SCM_INPUT_DEVICES_FWD_H_INCLUDED

#include <scm/core/memory.h>

namespace scm {
namespace inp {

class space_navigator;

typedef shared_ptr<space_navigator>         space_navigator_ptr;
typedef shared_ptr<space_navigator const>   space_navigator_cptr;

} // namespace inp
} // namespace scm

#endif // SCM_INPUT_DEVICES_FWD_H_INCLUDED
