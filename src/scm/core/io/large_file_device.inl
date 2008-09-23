
#include <scm/core/platform/platform.h>

#if    SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#include <scm/core/io/file_win.h>
#elif  SCM_PLATFORM == SCM_PLATFORM_LINUX
#error "atm unsupported platform"
#elif  SCM_PLATFORM == SCM_PLATFORM_APPLE
#error "atm unsupported platform"
#endif

namespace scm {
namespace io {

template <typename char_t>
large_file<char_t>::large_file()
{
#if    SCM_PLATFORM == SCM_PLATFORM_WINDOWS

    _impl.reset(new file_win());

#elif  SCM_PLATFORM == SCM_PLATFORM_LINUX
#error "atm unsupported platform"
#elif  SCM_PLATFORM == SCM_PLATFORM_APPLE
#error "atm unsupported platform"
#endif
}

template <typename char_t>
large_file<char_t>::large_file(const std::string&       file_path,
                               std::ios_base::openmode  open_mode,
                               bool                     disable_system_cache,
                               scm::uint32              read_write_buffer_size)
{
#if    SCM_PLATFORM == SCM_PLATFORM_WINDOWS

    _impl.reset(new file_win());

    open(file_path, open_mode, disable_system_cache, read_write_buffer_size);

#elif  SCM_PLATFORM == SCM_PLATFORM_LINUX
#error "atm unsupported platform"
#elif  SCM_PLATFORM == SCM_PLATFORM_APPLE
#error "atm unsupported platform"
#endif
}

template <typename char_t>
large_file<char_t>::large_file(const large_file<char_t>& rhs)
{
#if    SCM_PLATFORM == SCM_PLATFORM_WINDOWS

    _impl.reset(new file_win(*static_cast<file_win*>(rhs._impl.get())));

#elif  SCM_PLATFORM == SCM_PLATFORM_LINUX
#error "atm unsupported platform"
#elif  SCM_PLATFORM == SCM_PLATFORM_APPLE
#error "atm unsupported platform"
#endif
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
    file_base::char_type* char_buf = reinterpret_cast<file_base::char_type*>(s);

    return (_impl->read(char_buf, n));
}

template <typename char_t>
std::streamsize
large_file<char_t>::write(const char_type* s, std::streamsize n)
{
    const file_base::char_type* char_buf = reinterpret_cast<const file_base::char_type*>(s);

    return (_impl->write(char_buf, n));
}

template <typename char_t>
std::streampos
large_file<char_t>::seek(boost::iostreams::stream_offset    off,
                         std::ios_base::seek_dir            way)
{
    return (_impl->seek(off, way));
}

template <typename char_t>
void
large_file<char_t>::open(const std::string&         file_path,
                         std::ios_base::openmode    open_mode,
                         bool                       disable_system_cache,
                         scm::uint32                read_write_buffer_size)
{
    if (!_impl->open(file_path, open_mode, disable_system_cache, read_write_buffer_size)) {
        throw std::ios_base::failure("large_file<char_type>::open(): error opening file");
    }
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
