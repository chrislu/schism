
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_SYSTEM_INFO_H_INCLUDED
#define SCM_CORE_SYSTEM_INFO_H_INCLUDED

namespace scm {

bool is_host_little_endian();

template<typename T>
void swap_endian(T& val);

} // namespace scm

#include "system_info.inl"

#endif // SCM_CORE_SYSTEM_INFO_H_INCLUDED
