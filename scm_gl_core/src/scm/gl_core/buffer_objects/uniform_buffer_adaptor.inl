
#include <cassert>
#include <iostream>

#include <scm/gl_core/buffer_objects.h>
#include <scm/gl_core/render_device.h>

namespace scm {
namespace gl {

template <class host_block_type>
uniform_block<host_block_type>::uniform_block()
{
}

template <class host_block_type>
uniform_block<host_block_type>::uniform_block(const render_device_ptr& in_device)
  : _host_block(new host_block_type())
{
    _device_block = in_device->create_buffer(BIND_UNIFORM_BUFFER, USAGE_STREAM_DRAW, sizeof(host_block_type));
}

template <class host_block_type>
uniform_block<host_block_type>::uniform_block(const render_device_ptr& in_device, const host_block_type& in_block)
  : _host_block(new host_block_type(in_block))
{
    _device_block = in_device->create_buffer(BIND_UNIFORM_BUFFER, USAGE_STREAM_DRAW, sizeof(host_block_type));
    commit_block(in_device->main_context());
}

template <class host_block_type>
uniform_block<host_block_type>::~uniform_block()
{
    reset();
}

template <class host_block_type>
void
uniform_block<host_block_type>::reset()
{
    _device_block.reset();
    _host_block.reset();
}

template <class host_block_type>
void
uniform_block<host_block_type>::begin_manipulation(const render_context_ptr& in_context)
{
    assert(_device_block);
    assert(_host_block);

    _current_context = in_context;
}

template <class host_block_type>
void
uniform_block<host_block_type>::end_manipulation()
{
    assert(_device_block);
    assert(_host_block);

    commit_block(_current_context);
    _current_context.reset();
}

template <class host_block_type>
typename uniform_block<host_block_type>::block_type&
uniform_block<host_block_type>::operator*() const
{
    return (*_host_block);
}

template <class host_block_type>
typename uniform_block<host_block_type>::block_type*
uniform_block<host_block_type>::operator->() const
{
    return (_host_block.get());
}

template <class host_block_type>
typename uniform_block<host_block_type>::block_type*
uniform_block<host_block_type>::get_block() const
{
    return (_host_block.get());
}

template <class host_block_type>
const buffer_ptr&
uniform_block<host_block_type>::block_buffer() const
{
    return (_device_block);
}

template <class host_block_type>
void
uniform_block<host_block_type>::commit_block(const render_context_ptr& in_context)
{
    using namespace scm::gl;

    assert(_device_block);
    assert(_host_block);

    block_type* gpu_block = reinterpret_cast<block_type*>(in_context->map_buffer(_device_block, ACCESS_WRITE_INVALIDATE_BUFFER));

    if (memcpy(gpu_block, _host_block.get(), sizeof(block_type)) != gpu_block) {
        std::cerr << "uniform_block<>::commit_block(): error copying host block to gpu memory." << std::endl; 
    }

    in_context->unmap_buffer(_device_block);
}

template <class host_block_type>
uniform_block<host_block_type>
make_uniform_block(const render_device_ptr& in_device)
{
    return (uniform_block<host_block_type>(in_device));
}

template <class host_block_type>
uniform_block<host_block_type>
make_uniform_block(const render_device_ptr& in_device, const host_block_type& in_block)
{
    return (uniform_block<host_block_type>(in_device, in_block));
}

} // namespace gl
} // namespace scm
