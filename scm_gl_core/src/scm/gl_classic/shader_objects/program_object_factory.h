
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef PROGRAM_OBJECT_FACTORY_H_INCLUDED
#define PROGRAM_OBJECT_FACTORY_H_INCLUDED

#include <string>
#include <vector>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class program_object;

struct program_object_makefile
{
    // include files
    std::vector<std::string>        _vert_include_files;
    std::vector<std::string>        _frag_include_files;

    // source files
    std::string                     _vert_source_file;
    std::string                     _frag_source_file;

    // defines
    std::vector<std::string>        _vert_defines;
    std::vector<std::string>        _frag_defines;

}; // struct program_object_makefile

class __scm_export(gl_core) program_object_factory
{
public:
    static program_object       create(const program_object_makefile&    /*make_file*/);

    static bool                 output_available();
    static const std::string&   output();

protected:
    static void                 clear_output();

    static std::string          _output;
}; // class program_object_factory

} // namespace gl_classic
} // namespace scm

#endif // PROGRAM_OBJECT_FACTORY_H_INCLUDED
