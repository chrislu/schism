
#include <windows.h>

#include <cassert>

#include <boost/bind.hpp>
#include <boost/filesystem.hpp>

namespace scm {
namespace io {
namespace detail {

template <typename char_type>
scm::int64
large_file_device_windows<char_type>::file_size() const
{
    LARGE_INTEGER   cur_size_li;

    if (GetFileSizeEx(_file_handle, &cur_size_li) == 0) {
        throw std::ios_base::failure("large_file_device_windows<char_type>::seek(): error retrieving current file size");
    }

    return (static_cast<scm::int64>(cur_size_li.QuadPart));
}

template <typename char_type>
large_file_device_windows<char_type>::large_file_device_windows()
  : _file_handle(INVALID_HANDLE_VALUE),
    _current_position(0),
    _volume_sector_size(0),
    _read_write_buffer_size(0)
{
}

template <typename char_type>
large_file_device_windows<char_type>::large_file_device_windows(const large_file_device_windows<char_type>& rhs)
  : _file_handle(rhs._file_handle),
    _current_position(rhs._current_position),
    _volume_sector_size(rhs._volume_sector_size),
    _read_write_buffer_size(rhs._read_write_buffer_size),
    _read_write_buffer(rhs._read_write_buffer)
{
    // TODO initialize read write buffer if buiffer size > 0!
    // OR handle the rw buffer as a shared resource!
    // might be dangerous in threaded environments
    // to we flag this one here as not thread safe!
}

template <typename char_type>
large_file_device_windows<char_type>::~large_file_device_windows()
{
}

template <typename char_type>
std::streamsize
large_file_device_windows<char_type>::read(char_type* s, std::streamsize n)
{
    using namespace scm;

    // non system buffered read operation
    if (_read_write_buffer_size > 0) {
        int64           cur_file_size           = file_size();
        int64           read_iterations         = (n / _read_write_buffer_size) + (n % _read_write_buffer_size > 0 ? 1 : 0); // ceil
        int64           read_beg_file_offset    = (_current_position / _volume_sector_size) * _volume_sector_size;           // floor
        int64           write_beg_buffer_offset =  _current_position % _volume_sector_size;


        LARGE_INTEGER   cur_pos_li;
        cur_pos_li.QuadPart = read_beg_file_offset;
        SetFilePointer(_file_handle, cur_pos_li.LowPart, &cur_pos_li.HighPart, FILE_BEGIN);

        for (int64 i = 0; i < read_iterations; ++i) {
            int64       bytes_to_read           = 0;
            int64       bytes_read              = 0;
            
            if ((read_beg_file_offset + i * _read_write_buffer_size) < cur_file_size) {
                bytes_to_read = _read_write_buffer_size;
            }
            else {
                // current end of line
            }

            assert(bytes_to_read > 0);

            ReadFile(_file_handle, _read_write_buffer.get(), bytes_to_read, &bytes_read, 0);
        }
    }
    // normal system buffered operation
    else {
        // TODO
    }

    // TODO
    return (0);
}

template <typename char_type>
std::streamsize
large_file_device_windows<char_type>::write(const char_type* s, std::streamsize n)
{
    // TODO
    return (0);
}

template <typename char_type>
std::streampos
large_file_device_windows<char_type>::seek(boost::iostreams::stream_offset    off,
                                           std::ios_base::seek_dir            way)
{
    using namespace std;
    using namespace boost::iostreams;

    // determine current file size
    scm::int64      cur_file_size = file_size();

    // determine new value of _current_position
    stream_offset   next_pos;

    if (way == ios_base::beg) {
        next_pos = off;
    } else if (way == ios_base::cur) {
        next_pos = _current_position + off;
    } else if (way == ios_base::end) {
        next_pos = cur_file_size + off;
    } else {
        throw std::ios_base::failure("large_file_device_windows<char_type>::seek(): bad seek direction");
    }

    // check for errors
    if (next_pos < 0 || next_pos > cur_file_size) {
        throw std::ios_base::failure("large_file_device_windows<char_type>::seek(): bad seek offset");
    }

    _current_position = next_pos;

    return (_current_position);
}

template <typename char_type>
void
large_file_device_windows<char_type>::open(const std::string&         file_path,
                                           std::ios_base::openmode    open_mode,
                                           bool                       disable_system_cache,
                                           scm::uint32                read_write_buffer_size)
{
    using namespace boost::filesystem;

    path            input_file_path(file_path, native);
    path            complete_input_file_path(system_complete(input_file_path));

    bool            input_file_exists = exists(complete_input_file_path);
    std::string     input_root_path = complete_input_file_path.root_name();

    // translate open mode to access mode
    DWORD desired_access = 0;
    DWORD read_write_buffer_access = 0;

    if (open_mode & std::ios_base::in) {
        desired_access              |= GENERIC_READ;
        read_write_buffer_access    =  PAGE_READONLY;
    }
    if (open_mode & std::ios_base::out) {
        desired_access              |= GENERIC_WRITE;
        read_write_buffer_access    =  PAGE_READWRITE;
    }

    // share mode
    DWORD share_mode = 0;

    // translate open mode to creation modes
    DWORD creation_disposition = 0;

    if (input_file_exists) {
        if (    (open_mode & std::ios_base::out
              || open_mode & std::ios_base::in)
            && !(open_mode & std::ios_base::trunc)) {
            creation_disposition = OPEN_ALWAYS;
        }
        else if (   open_mode & std::ios_base::out
                 && open_mode & std::ios_base::trunc) {
            creation_disposition = TRUNCATE_EXISTING;
        }
        else {
            throw std::ios_base::failure("large_file_device_windows<char_type>::open(): illegal open mode");
        }
    }
    else {
        if (    (open_mode & std::ios_base::out
              || open_mode & std::ios_base::in)
            && !(open_mode & std::ios_base::trunc)) {
            creation_disposition = CREATE_NEW;
        }
        else if (   open_mode & std::ios_base::out
                 && open_mode & std::ios_base::trunc) {
            creation_disposition = CREATE_NEW;
        }
        else {
            throw std::ios_base::failure("large_file_device_windows<char_type>::open(): illegal open mode");
        }
    }

    // file attributes
    DWORD flags_and_attributes = FILE_ATTRIBUTE_NORMAL;

    if (disable_system_cache) {
        flags_and_attributes |= FILE_FLAG_NO_BUFFERING;

        // retrieve the sector size information
        DWORD   sectors_per_cluster;
        DWORD   bytes_per_sector;
        DWORD   free_clusters;
        DWORD   total_clusters;

        if (GetDiskFreeSpace(input_root_path.c_str(),
                             &sectors_per_cluster,
                             &bytes_per_sector,
                             &free_clusters,
                             &total_clusters) == FALSE) {
            throw std::ios_base::failure("large_file_device_windows<char_type>::open(): error retrieving sector size information");
        }

        _volume_sector_size = bytes_per_sector;

        assert(_volume_sector_size != 0);

        // calculate the correct read write buffer size (round up to full multiple of bytes per sector)
        _read_write_buffer_size = (  (read_write_buffer_size / _volume_sector_size)
                                   + (read_write_buffer_size % _volume_sector_size > 0 ? 1 : 0)) * _volume_sector_size;

        assert(_read_write_buffer_size % _volume_sector_size == 0);
        assert(read_write_buffer_access != 0);

        // allocate read write buffer using VirtualAlloc
        // this aligns the memory region to page sizes
        // this memory has to be deallocated using VirtualFree, so the deallocator to the smart pointer
        _read_write_buffer.reset(static_cast<scm::uint8*>(VirtualAlloc(0, _read_write_buffer_size, MEM_COMMIT | MEM_RESERVE, read_write_buffer_access)),
                                                          boost::bind(VirtualFree, _1, 0, MEM_RELEASE));

        if (!_read_write_buffer) {
            throw std::ios_base::failure(  std::string("large_file_device_windows<char_type>::open(): error allocating read write buffer for no system buffering operation: ")
                                         + complete_input_file_path.string());
        }
    }

    _file_handle = CreateFile(complete_input_file_path.string().c_str(),
                              desired_access,
                              share_mode,
                              0,
                              creation_disposition,
                              flags_and_attributes,
                              0);

    if (_file_handle == INVALID_HANDLE_VALUE) {
        throw std::ios_base::failure(  std::string("large_file_device_windows<char_type>::open(): error creating/opening file ")
                                     + complete_input_file_path.string());
    }
}

template <typename char_type>
bool
large_file_device_windows<char_type>::is_open() const
{
    return (_file_handle != INVALID_HANDLE_VALUE);
}

template <typename char_type>
void
large_file_device_windows<char_type>::close()
{
    if (is_open()) {
        CloseHandle(_file_handle);
        _file_handle = INVALID_HANDLE_VALUE;

        // reset read write buffer invoking VirtualFree
        _read_write_buffer.reset();
    }
}

} // namespace detail
} // namespace io
} // namespace scm
