
#include <windows.h>

#include <cassert>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

namespace scm {
namespace io {
namespace detail {

template <typename char_type>
scm::int64
large_file_device_windows<char_type>::floor_vss(const scm::int64 in_val) const
{
    return((in_val / _volume_sector_size) * _volume_sector_size);
}
template <typename char_type>
scm::int64
large_file_device_windows<char_type>::ceil_vss(const scm::int64 in_val) const
{
    return ( ((in_val / _volume_sector_size)
            + (in_val % _volume_sector_size > 0 ? 1 : 0)) * _volume_sector_size);
}

template <typename char_type>
scm::int64
large_file_device_windows<char_type>::file_size() const
{
    assert(_file_handle != INVALID_HANDLE_VALUE);

    LARGE_INTEGER   cur_size_li;

    if (GetFileSizeEx(_file_handle, &cur_size_li) == 0) {
        throw std::ios_base::failure("large_file_device_windows<char_type>::seek(): error retrieving current file size");
    }

    return (static_cast<scm::int64>(cur_size_li.QuadPart));
}

template <typename char_type>
bool
large_file_device_windows<char_type>::set_file_pointer(scm::int64 new_pos)
{
    assert(_file_handle != INVALID_HANDLE_VALUE);

    LARGE_INTEGER   position_li;

    position_li.QuadPart = new_pos;

    if (   SetFilePointer(_file_handle, position_li.LowPart, &position_li.HighPart, FILE_BEGIN) == INVALID_SET_FILE_POINTER
        && GetLastError() != NO_ERROR) {
        return (false);
    }

    return (true);
}

template <typename char_type>
large_file_device_windows<char_type>::large_file_device_windows()
  : _file_handle(INVALID_HANDLE_VALUE),
    _file_size(0),
    _current_position(0),
    _open_mode(0),
    _volume_sector_size(0),
    _rw_buffer_size(0),
    _rw_buffered_start(0),
    _rw_buffered_end(0)
{
}

template <typename char_type>
large_file_device_windows<char_type>::large_file_device_windows(const large_file_device_windows<char_type>& rhs)
  : _file_path(rhs._file_path),
    _file_handle(rhs._file_handle),
    _file_size(rhs._file_size),
    _current_position(rhs._current_position),
    _open_mode(rhs._open_mode),
    _volume_sector_size(rhs._volume_sector_size),
    _rw_buffer_size(rhs._rw_buffer_size),
    _rw_buffer(rhs._rw_buffer),
    _rw_buffered_start(rhs._rw_buffered_start),
    _rw_buffered_end(rhs._rw_buffered_end)
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
large_file_device_windows<char_type>::read(char_type*       output_buffer,
                                           std::streamsize  num_bytes_to_read)
{
    assert(_file_handle != INVALID_HANDLE_VALUE);

    using namespace scm;

    uint8*      output_byte_buffer  = reinterpret_cast<uint8*>(output_buffer);
    int64       bytes_read          = 0;

    // non system buffered read operation
    if (_rw_buffer_size > 0) {

        // set read pointer to beginning
        int64           current_position_vss    = 0;
        int64           file_size_vss           = 0;

        // align current position to volume sector size
        current_position_vss    = floor_vss(_current_position);
        file_size_vss           = ceil_vss(_file_size);

        // calculate the bytes to read in multiples of the volume sector size
        int64   bytes_to_read_vss   = 0;

        bytes_to_read_vss  = ceil_vss(math::min((_current_position % _volume_sector_size) + num_bytes_to_read,
                                                _file_size - current_position_vss));

        while (bytes_to_read_vss > 0) {
            int64   buffer_bytes_to_read        = 0;
            int64   end_position_vss            = 0;
            int64   rw_buffer_read_offset       = 0;
            DWORD   buffer_bytes_read           = 0;

            // determine the amount of bytes to read to the buffer
            // note if not a complete buffer is filled, the remainder is still
            // a interger multiple of the volume sector size
            buffer_bytes_to_read        = math::clamp<int64>(bytes_to_read_vss, 0, _rw_buffer_size);
            end_position_vss            = current_position_vss + buffer_bytes_to_read;

            assert(buffer_bytes_to_read > 0);
            assert(_rw_buffer);

            bool read_beg_in_rw_buf =    current_position_vss >= _rw_buffered_start
                                      && current_position_vss <  _rw_buffered_end;
            bool read_end_in_rw_buf =    end_position_vss >  _rw_buffered_start
                                      && end_position_vss <= _rw_buffered_end;

            // test if read begin or end is in buffered range
            if (read_beg_in_rw_buf && read_end_in_rw_buf) {
                // we can read this stuff completely from the rw buffer
                buffer_bytes_read       = end_position_vss - current_position_vss;
                rw_buffer_read_offset   = current_position_vss - _rw_buffered_start;

                // the rw buffer is left untouched
            }
            else if (!read_beg_in_rw_buf && read_end_in_rw_buf) {
                // the end position is in the buffer
                int64   bytes_cached        = end_position_vss - _rw_buffered_start;
                int64   bytes_left_to_read  = math::max(_rw_buffered_start - current_position_vss,
                                                        math::min(_rw_buffered_start, _rw_buffer_size - bytes_cached));
                int64   new_read_pos_vss    = _rw_buffered_start - bytes_left_to_read;

                // copy the buffered to the end
                memmove(_rw_buffer.get() + bytes_left_to_read,
                        _rw_buffer.get(),
                        bytes_cached);

                if (!set_file_pointer(new_read_pos_vss)) {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::read(): unable to set file pointer to current position");
                }

                // read the necessary data from disc
                if (ReadFile(_file_handle,
                             _rw_buffer.get(),
                             bytes_left_to_read,
                             &buffer_bytes_read,
                             0) == 0)
                {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::read(): error reading from file");
                }

                if (buffer_bytes_read != bytes_left_to_read) {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::read(): error reading from file");
                }

                // set values for the copy operation
                buffer_bytes_read       = buffer_bytes_read + bytes_cached;
                rw_buffer_read_offset   = bytes_left_to_read - (_rw_buffered_start - current_position_vss);

                // set the rw buffer region
                _rw_buffered_start = new_read_pos_vss;
                _rw_buffered_end   = _rw_buffered_start + ceil_vss(buffer_bytes_read);
            }
            else if (read_beg_in_rw_buf && !read_end_in_rw_buf) {
                // beginning position in the buffer
                int64   bytes_cached        = _rw_buffered_end - current_position_vss;
                int64   bytes_left_to_read  = math::max(end_position_vss - _rw_buffered_end,
                                                        math::min(file_size_vss - _rw_buffered_end,
                                                                  _rw_buffer_size - bytes_cached));
                int64   new_read_pos_vss    = current_position_vss + bytes_cached;

                // copy the buffered to the end
                memmove(_rw_buffer.get(),
                        _rw_buffer.get() + (current_position_vss - _rw_buffered_start),
                        bytes_cached);

                if (!set_file_pointer(new_read_pos_vss)) {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::read(): unable to set file pointer to current position");
                }

                // read the necessary data from disc
                if (ReadFile(_file_handle,
                             _rw_buffer.get() + bytes_cached,
                             bytes_left_to_read,
                             &buffer_bytes_read,
                             0) == 0)
                {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::read(): error reading from file");
                }

                // set values for the copy operation
                buffer_bytes_read       = buffer_bytes_read + bytes_cached;
                rw_buffer_read_offset   = 0;

                // set the rw buffer region
                _rw_buffered_start = new_read_pos_vss - bytes_cached;
                _rw_buffered_end   = _rw_buffered_start + ceil_vss(buffer_bytes_read);
            }
            else {
                // now we know we have to read everything from disk so fill up our internal rw buffer completely...
                if (!set_file_pointer(current_position_vss)) {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::read(): unable to set file pointer to current position");
                }

                if (ReadFile(_file_handle,
                             _rw_buffer.get(),
                             _rw_buffer_size,
                             &buffer_bytes_read,
                             0) == 0)
                {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::read(): error reading from file");
                }

                // set the rw buffer region
                _rw_buffered_start = current_position_vss;
                _rw_buffered_end   = _rw_buffered_start + ceil_vss(buffer_bytes_read);
            }

            // check if eof reached
            if (buffer_bytes_read <= 0) {
                return ((bytes_read == 0) ? -1 : bytes_read);
            }

            // copy data from rw buffer to output
            int64   buf_read_offset     = (_current_position % _volume_sector_size) + rw_buffer_read_offset;
            int64   buf_read_length     = math::min(math::max<int64>(0, static_cast<int64>(buffer_bytes_read) - (buf_read_offset - rw_buffer_read_offset)),
                                          math::min(num_bytes_to_read - bytes_read,
                                                    _rw_buffer_size - buf_read_offset));

            assert(buf_read_length >= 0);
            assert(buf_read_offset >= 0);
            assert(buf_read_offset + buf_read_length <= _rw_buffer_size);
            assert(buf_read_length <= num_bytes_to_read);

            CopyMemory(output_byte_buffer,
                       _rw_buffer.get() + buf_read_offset,
                       buf_read_length);

            bytes_to_read_vss        = (buffer_bytes_read < buffer_bytes_to_read) ? 0 : bytes_to_read_vss - buffer_bytes_read;
            output_byte_buffer      += buf_read_length;
            _current_position       += buf_read_length;
            current_position_vss    += buffer_bytes_read;
            bytes_read              += buf_read_length;
        }
    }
    // normal system buffered operation
    else {
        if (!set_file_pointer(_current_position)) {
            throw std::ios_base::failure("large_file_device_windows<char_type>::read(): unable to set file pointer to current position");
        }

        DWORD   file_bytes_read       = 0;

        if (ReadFile(_file_handle, output_byte_buffer, num_bytes_to_read, &file_bytes_read, 0) == 0) {
            throw std::ios_base::failure("large_file_device_windows<char_type>::read(): error reading from file");
        }

        if (file_bytes_read == 0) {
            // eof
            return (-1);
        }
        if (file_bytes_read <= num_bytes_to_read) {
            _current_position   += file_bytes_read;
            bytes_read           = file_bytes_read;
        }
        else {
            throw std::ios_base::failure("large_file_device_windows<char_type>::read(): unknown error reading from file");
        }
    }

    assert(bytes_read > 0);

    return (bytes_read);
}

template <typename char_type>
std::streamsize
large_file_device_windows<char_type>::write(const char_type*    input_buffer,
                                            std::streamsize     num_bytes_to_write)
{
    assert(_file_handle != INVALID_HANDLE_VALUE);

    using namespace scm;

    if (_open_mode & std::ios_base::app) {
        _current_position = _file_size;
    }

    const uint8*    input_byte_buffer   = reinterpret_cast<const uint8*>(input_buffer);
    int64           bytes_written       = 0;

    // non system buffered read operation
    if (_rw_buffer_size > 0) {

        // set read pointer to beginning
        int64           write_beg_file_offset_vss;
        int64           write_sector_prefetch_size;

        write_beg_file_offset_vss       = (_current_position / _volume_sector_size) * _volume_sector_size;  // floor
        write_sector_prefetch_size      = _current_position % _volume_sector_size;

        // set file pointer to beginning of volume sector
        if (!set_file_pointer(write_beg_file_offset_vss)) {
            throw std::ios_base::failure("large_file_device_windows<char_type>::write(): unable to set file pointer to current position");
        }

        if (write_sector_prefetch_size > 0) {
            if (_current_position <= _file_size) {
                int64 current_pos = _current_position;

                seek(write_beg_file_offset_vss, std::ios_base::beg);

                // ok we need some data from the beginning of the sector
                if (read(_rw_buffer.get(), write_sector_prefetch_size) < write_sector_prefetch_size) {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::write(): unable read data from beginning of sector");
                }

                seek(current_pos, std::ios_base::beg);

                // reset the file pointer to the beginning of the volume sector
                if (!set_file_pointer(write_beg_file_offset_vss)) {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::write(): unable to set file pointer to current position");
                }
            }
            else {
                if (SetEndOfFile(_file_handle) == 0) {
                    throw std::ios_base::failure("large_file_device_windows<char_type>::write(): unable to set new end of file");
                }
            }
        }

        // calculate the bytes to write in multiples of the volume sector size
        int64   bytes_to_write_vss  = 0;

        bytes_to_write_vss = write_sector_prefetch_size + num_bytes_to_write;
        bytes_to_write_vss =  ((bytes_to_write_vss / _volume_sector_size)
                             + (bytes_to_write_vss % _volume_sector_size > 0 ? 1 : 0)) * _volume_sector_size; // ceil

        while (bytes_to_write_vss > 0) {
            int64   buffer_bytes_to_write   = 0;
            int64   buffer_fill_start_off   = 0;
            int64   buffer_fill_length      = 0;
            DWORD   buffer_bytes_written    = 0;

            // determine the amount of bytes to write to the buffer
            // note if not a complete buffer is filled, the remainder is still
            // a interger multiple of the volume sector size
            buffer_bytes_to_write = math::clamp<int64>(bytes_to_write_vss, 0, _rw_buffer_size);
            buffer_fill_start_off = _current_position % _volume_sector_size;
            buffer_fill_length    = math::min(buffer_bytes_to_write - buffer_fill_start_off,
                                              num_bytes_to_write - bytes_written);

            assert(buffer_bytes_to_write > 0);
            assert(_rw_buffer);

            // fill in the data to be written
            CopyMemory(_rw_buffer.get() + buffer_fill_start_off,
                       input_byte_buffer,
                       buffer_fill_length);

            if (WriteFile(_file_handle, _rw_buffer.get(), buffer_bytes_to_write, &buffer_bytes_written, 0) == 0) {
                throw std::ios_base::failure("large_file_device_windows<char_type>::write(): error writing to file");
            }

            bytes_to_write_vss  -= buffer_bytes_to_write;
            input_byte_buffer   += buffer_fill_length;
            _current_position   += buffer_fill_length;
            bytes_written       += buffer_fill_length;
        }

        // ok now we need to know the real file size, not the size rounded
        // to integer multiple of the volume sector size at the end
        if (_file_size < _current_position) {
            _file_size = _current_position;
        }
    }
    // normal system buffered operation
    else {
        if (!set_file_pointer(_current_position)) {
            throw std::ios_base::failure("large_file_device_windows<char_type>::write(): unable to set file pointer to current position");
        }

        DWORD   file_bytes_written  = 0;

        if (WriteFile(_file_handle, input_byte_buffer, num_bytes_to_write, &file_bytes_written, 0) == 0) {
            throw std::ios_base::failure("large_file_device_windows<char_type>::write(): error writing from file");
        }

        if (file_bytes_written <= num_bytes_to_write) {
            _current_position   += file_bytes_written;
            bytes_written        = file_bytes_written;
        }
        else {
            throw std::ios_base::failure("large_file_device_windows<char_type>::write(): unknown error reading from file");
        }
    }

    return (bytes_written);
}

template <typename char_type>
std::streampos
large_file_device_windows<char_type>::seek(boost::iostreams::stream_offset    off,
                                           std::ios_base::seek_dir            way)
{
    using namespace std;
    using namespace boost::iostreams;

    // determine current file size
    scm::int64      cur_file_size = _file_size;

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
    if (next_pos < 0 /*|| next_pos > cur_file_size*/) {
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
        read_write_buffer_access    =  PAGE_READWRITE;
    }
    if (open_mode & std::ios_base::out) {
        desired_access              |= GENERIC_WRITE;
        read_write_buffer_access    =  PAGE_READWRITE;
    }

    // share mode
    DWORD share_mode = FILE_SHARE_READ;

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
        _rw_buffer_size = ceil_vss(read_write_buffer_size);;

        assert(_rw_buffer_size % _volume_sector_size == 0);
        assert(read_write_buffer_access != 0);

        // allocate read write buffer using VirtualAlloc
        // this aligns the memory region to page sizes
        // this memory has to be deallocated using VirtualFree, so the deallocator to the smart pointer
        _rw_buffer.reset(static_cast<char*>(VirtualAlloc(0, _rw_buffer_size, MEM_COMMIT | MEM_RESERVE, read_write_buffer_access)),
                                            boost::bind(VirtualFree, _1, 0, MEM_RELEASE));

        if (!_rw_buffer) {
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

    if (   open_mode & std::ios_base::ate
        || open_mode & std::ios_base::app) {

        _current_position = file_size();
    }

    _file_path  = complete_input_file_path.string();
    _open_mode  = open_mode;
    _file_size  = file_size();
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
        // if we were non system buffered, so i may be possible to
        // be too large because of volume sector size restrictions
        if (   _rw_buffer_size
            && _open_mode & std::ios_base::out) {
            if (_file_size != file_size()) {

            CloseHandle(_file_handle);

            _file_handle = CreateFile(_file_path.c_str(),
                                      GENERIC_WRITE,
                                      PAGE_READONLY,
                                      0,
                                      OPEN_EXISTING,
                                      FILE_ATTRIBUTE_NORMAL,
                                      0);

            }
            if (_file_handle == INVALID_HANDLE_VALUE) {
                throw std::ios_base::failure(  std::string("large_file_device_windows<char_type>::close(): error opening file for truncating end: ")
                                             + _file_path);
            }
            set_file_pointer(_file_size);
            if (SetEndOfFile(_file_handle) == 0) {
                throw std::ios_base::failure(  std::string("large_file_device_windows<char_type>::close(): error truncating end of file: ")
                                             + _file_path);
            }
        }
        CloseHandle(_file_handle);
        _file_handle = INVALID_HANDLE_VALUE;
    }

    // reset read write buffer invoking VirtualFree
    _file_path              = std::string("");
    _file_handle            = INVALID_HANDLE_VALUE;
    _file_size              = 0;
    _volume_sector_size     = 0;
    _rw_buffer.reset();
    _rw_buffer_size = 0;
    _open_mode              = 0;
    _current_position       = 0;
}

template <typename char_type>
std::streamsize
large_file_device_windows<char_type>::optimal_buffer_size() const
{
    if (_rw_buffer_size && _volume_sector_size) {
        return (scm::math::max<scm::int32>(_volume_sector_size * 4, _rw_buffer_size / 16));
    }
    else {
        return (4096u);
    }
}

} // namespace detail
} // namespace io
} // namespace scm
