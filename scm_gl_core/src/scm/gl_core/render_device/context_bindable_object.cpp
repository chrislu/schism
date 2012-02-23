
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "context_bindable_object.h"

namespace scm {
namespace gl {

context_bindable_object::context_bindable_object()
  : _gl_object_id(0)
  , _gl_object_target(0)
  , _gl_object_binding(0)
{
}

context_bindable_object::~context_bindable_object()
{
}

} // namespace gl
} // namespace scm
