
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_UNIFORM_BUFFER_ADAPTOR_H_INCLUDED
#define SCM_GL_CORE_UNIFORM_BUFFER_ADAPTOR_H_INCLUDED

#include <scm/core/memory.h>

#include <scm/gl_core/buffer_objects/buffer_objects_fwd.h>
#include <scm/gl_core/render_device/render_device_fwd.h>

namespace scm {
namespace gl {

// uniform_block //////////////////////////////////////////////////////////////////////////////////
template <class host_block_type>
class uniform_block
{
public:
    typedef host_block_type                                  block_type;
    typedef shared_ptr<uniform_block<host_block_type> >      ptr;
    typedef shared_ptr<uniform_block<host_block_type> const> cptr;

public:
    uniform_block();
    uniform_block(const render_device_ptr& in_device);
    uniform_block(const render_device_ptr& in_device, const host_block_type& in_block);
    ~uniform_block();

    void                        reset();

    void                        begin_manipulation(const render_context_ptr& in_context);
    void                        end_manipulation();

    block_type&                 operator*() const;
    block_type*                 operator->() const;
    block_type*                 get_block() const;

    const buffer_ptr&           block_buffer() const;

protected:
    void                        commit_block(const render_context_ptr& in_context);

protected:
    shared_ptr<block_type>      _host_block;
    buffer_ptr                  _device_block;
    render_context_ptr          _current_context;

}; // class uniform_block

template <class host_block_type>
uniform_block<host_block_type>
make_uniform_block(const render_device_ptr& in_device);

template <class host_block_type>
uniform_block<host_block_type>
make_uniform_block(const render_device_ptr& in_device, const host_block_type& in_block);
// end uniform_block //////////////////////////////////////////////////////////////////////////////

// uniform_block_array ////////////////////////////////////////////////////////////////////////////
template <class host_block_type>
class uniform_block_array
{
public:
    typedef host_block_type                                        block_type;
    typedef shared_ptr<uniform_block_array<host_block_type> >      ptr;
    typedef shared_ptr<uniform_block_array<host_block_type> const> cptr;

public:
    uniform_block_array();
    uniform_block_array(const render_device_ptr& in_device, const scm::size_t in_array_size);
    ~uniform_block_array();

    void                        reset();

    void                        begin_manipulation(const render_context_ptr& in_context);
    void                        end_manipulation();

    block_type&                 operator[](const scm::size_t in_index) const;
    block_type*                 get_block(const scm::size_t in_index) const;
    scm::size_t                 block_offset(const scm::size_t in_index) const;

    const buffer_ptr&           block_buffer() const;
    const scm::size_t           array_size() const;

protected:
    void                        commit_block(const render_context_ptr& in_context);

protected:
    shared_array<block_type>    _host_block;
    buffer_ptr                  _device_block;
    scm::size_t                 _array_size;
    scm::size_t                 _array_element_alignment;

    render_context_ptr          _current_context;

}; // class uniform_block_array

template <class host_block_type>
uniform_block_array<host_block_type>
make_uniform_block_array(const render_device_ptr& in_device, const scm::size_t in_array_size);
// end uniform_block_array ////////////////////////////////////////////////////////////////////////

} // namespace gl
} // namespace scm

#include "uniform_buffer_adaptor.inl"

#endif // SCM_GL_CORE_UNIFORM_BUFFER_ADAPTOR_H_INCLUDED
