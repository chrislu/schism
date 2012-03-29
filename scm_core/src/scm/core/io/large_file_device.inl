
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <scm/core/platform/platform.h>


namespace scm {
namespace io {

template <typename char_t>
large_file<char_t>::large_file()
  : _position(0)
{
    _impl.reset(new file());
}

template <typename char_t>
large_file<char_t>::large_file(const std::string&       file_path,
                               std::ios_base::openmode  open_mode,
                               bool                     disable_system_cache,
                               scm::uint32              read_write_buffer_size,
                               scm::uint32              read_write_asynchronous_requests)
{
    _impl.reset(new file());

    open(file_path, open_mode, disable_system_cache, read_write_buffer_size, read_write_asynchronous_requests);
}

template <typename char_t>
large_file<char_t>::large_file(const large_file<char_t>& rhs)
{
    _impl.reset(new file(*rhs._impl.get()));
}

template <typename char_t>
large_file<char_t>::~large_file()
{
    _impl.reset();
}

template <typename char_t>
std::streamsize
large_file<char_t>::read(char_type* s, std::streamsize n)
{
    file::char_type* char_buf = reinterpret_cast<file::char_type*>(s);
    file::size_type r = _impl->read(char_buf, _position, n);
    _position += r;

    return r;
}

template <typename char_t>
std::streamsize
large_file<char_t>::write(const char_type* s, std::streamsize n)
{
    const file::char_type* char_buf = reinterpret_cast<const file::char_type*>(s);
    file::size_type r = _impl->write(char_buf, _position, n);
    _position += r;

    return r;
}

template <typename char_t>
std::streampos
large_file<char_t>::seek(boost::iostreams::stream_offset    off,
                         std::ios_base::seek_dir            way)
{
    _position = _impl->seek(off, way);
    return _position;
}

template <typename char_t>
void
large_file<char_t>::open(const std::string&         file_path,
                         std::ios_base::openmode    open_mode,
                         bool                       disable_system_cache,
                         scm::uint32                read_write_buffer_size,
                         scm::uint32                read_write_asynchronous_requests)
{
    if (!_impl->open(file_path, open_mode, disable_system_cache, read_write_buffer_size, read_write_asynchronous_requests)) {
        throw std::ios_base::failure("large_file<char_type>::open(): error opening file");
    }
    _position = 0;
}

template <typename char_t>
bool
large_file<char_t>::is_open() const
{
    return (_impl->is_open());
}

template <typename char_t>
void
large_file<char_t>::close()
{
    _impl->close();
}

template <typename char_t>
std::streamsize
large_file<char_t>::optimal_buffer_size() const 
{
    return (_impl->optimal_buffer_size());
}

} // namespace io
} // namespace scm
