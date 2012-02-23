
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_SHADER_OBJECTS_FWD_H_INCLUDED
#define SCM_GL_CORE_SHADER_OBJECTS_FWD_H_INCLUDED

#include <list>
#include <string>

#include <scm/core/memory.h>

namespace scm {
namespace gl {

class shader;
class program;
class uniform_base;

class shader_macro;
class shader_macro_array;

class stream_capture;
class stream_capture_array;
class separate_stream_capture;
class interleaved_capture;

typedef std::list<std::string>          shader_include_path_list;

typedef shared_ptr<shader>              shader_ptr;
typedef shared_ptr<const shader>        shader_cptr;
typedef shared_ptr<program>             program_ptr;
typedef shared_ptr<const program>       program_cptr;
typedef weak_ptr<program>               program_wtr;
typedef weak_ptr<const program>         program_cwtr;

typedef shared_ptr<uniform_base>        uniform_ptr;
typedef shared_ptr<const uniform_base>  uniform_cptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_SHADER_OBJECTS_FWD_H_INCLUDED
