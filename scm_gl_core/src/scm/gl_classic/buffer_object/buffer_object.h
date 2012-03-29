
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_OGL_BUFFER_OBJECT_H_INCLUDED
#define SCM_OGL_BUFFER_OBJECT_H_INCLUDED

#include <cstddef>

#include <boost/noncopyable.hpp>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class __scm_export(gl_core) buffer_object : boost::noncopyable
{
public:
    class binding_guard : boost::noncopyable
    {
    public:
        binding_guard(unsigned /*target*/, unsigned /*binding*/);
        virtual ~binding_guard();
    private:
        int             _save;
        unsigned        _binding;
        unsigned        _target;
    };

public:
    buffer_object(unsigned /*target*/, unsigned /*binding*/);
    virtual ~buffer_object();

    void                bind() const;
    void                unbind() const;

    bool                reset();
    void                clear();

    bool                buffer_data(std::size_t /*size*/, const void* /*data*/, unsigned /*usage*/);

protected:
    bool                generate_buffer();
    void                delete_buffer();

    
    unsigned            _id;
    unsigned            _target;
    unsigned            _target_binding;

private:

};

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_OGL_BUFFER_OBJECT_H_INCLUDED
