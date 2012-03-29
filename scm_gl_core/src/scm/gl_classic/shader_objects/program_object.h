
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef PROGRAM_OBJECT_H_INCLUDED
#define PROGRAM_OBJECT_H_INCLUDED

#include <string>

#include <boost/noncopyable.hpp>

#include <scm/core/pointer_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class shader_object;

class __scm_export(gl_core) program_object
{
public:
    class binding_guard : boost::noncopyable
    {
    public:
        binding_guard();
        virtual ~binding_guard();
    private:
        int             _save_current_program;
    };

public:
    program_object();
    program_object(const program_object& prog_obj);
    virtual ~program_object();

    program_object& operator=(const program_object& rhs);

    bool                attach_shader(const gl_classic::shader_object&);
    bool                link();
    bool                validate();
    void                bind() const;
    void                unbind() const;

    unsigned int        program_id() const;

    bool                ok() const {return (_ok); }
    const bool          linker_output_available() const { return (!_linker_out.empty()); }
    const std::string&  linker_output() const { return (_linker_out); }
    const std::string&  valitation_output() const { return (_validate_out); }

    void                uniform_1f(const std::string&, float) const;
    void                uniform_2f(const std::string&, float, float) const;
    void                uniform_3f(const std::string&, float, float, float) const;
    void                uniform_4f(const std::string&, float, float, float, float) const;

    void                uniform_1fv(const std::string&, unsigned int, const float*) const;
    void                uniform_2fv(const std::string&, unsigned int, const float*) const;
    void                uniform_3fv(const std::string&, unsigned int, const float*) const;
    void                uniform_4fv(const std::string&, unsigned int, const float*) const;

    void                uniform_1i(const std::string&, int) const;
    void                uniform_2i(const std::string&, int, int) const;
    void                uniform_3i(const std::string&, int, int, int) const;
    void                uniform_4i(const std::string&, int, int, int, int) const;

    void                uniform_1iv(const std::string&, unsigned int, const int*) const;
    void                uniform_2iv(const std::string&, unsigned int, const int*) const;
    void                uniform_3iv(const std::string&, unsigned int, const int*) const;
    void                uniform_4iv(const std::string&, unsigned int, const int*) const;

    void                uniform_1ui(const std::string&, unsigned int) const;
    void                uniform_2ui(const std::string&, unsigned int, unsigned int) const;
    void                uniform_3ui(const std::string&, unsigned int, unsigned int, unsigned int) const;
    void                uniform_4ui(const std::string&, unsigned int, unsigned int, unsigned int, unsigned int) const;

    void                uniform_1uiv(const std::string&, unsigned int, const unsigned int*) const;
    void                uniform_2uiv(const std::string&, unsigned int, const unsigned int*) const;
    void                uniform_3uiv(const std::string&, unsigned int, const unsigned int*) const;
    void                uniform_4uiv(const std::string&, unsigned int, const unsigned int*) const;

    void                uniform_matrix_2fv(const std::string&, unsigned int, bool, const float*) const;
    void                uniform_matrix_3fv(const std::string&, unsigned int, bool, const float*) const;
    void                uniform_matrix_4fv(const std::string&, unsigned int, bool, const float*) const;

protected:
private:
    int                 uniform_location(const std::string&) const;

    scm::shared_ptr<unsigned int>   _prog;

    bool                            _ok;

    std::string                     _linker_out;
    std::string                     _validate_out;

}; // class program_object

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // PROGRAM_OBJECT_H_INCLUDED
