
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_LOG_CONSOLE_COLOR_H_INCLUDED
#define SCM_CORE_LOG_CONSOLE_COLOR_H_INCLUDED

#include <iostream>
#include <iomanip>

#include <scm/core/platform/platform.h>
#include <scm/core/platform/windows.h>

namespace scm {
namespace util {

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS
class default_console
{
public:
    static const WORD _bg_mask = BACKGROUND_BLUE | BACKGROUND_GREEN | BACKGROUND_RED | BACKGROUND_INTENSITY;
    static const WORD _fg_mask = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY;

    static const WORD _fg_black          = 0;
    static const WORD _fg_low_red        = FOREGROUND_RED;
    static const WORD _fg_low_green      = FOREGROUND_GREEN;
    static const WORD _fg_low_blue       = FOREGROUND_BLUE;
    static const WORD _fg_low_cyan       = _fg_low_green   | _fg_low_blue;
    static const WORD _fg_low_magenta    = _fg_low_red     | _fg_low_blue;
    static const WORD _fg_low_yellow     = _fg_low_red     | _fg_low_green;
    static const WORD _fg_low_white      = _fg_low_red     | _fg_low_green | _fg_low_blue;
    static const WORD _fg_gray           = _fg_black       | FOREGROUND_INTENSITY; 
    static const WORD _fg_hi_white       = _fg_low_white   | FOREGROUND_INTENSITY; 
    static const WORD _fg_hi_blue        = _fg_low_blue    | FOREGROUND_INTENSITY; 
    static const WORD _fg_hi_green       = _fg_low_green   | FOREGROUND_INTENSITY; 
    static const WORD _fg_hi_red         = _fg_low_red     | FOREGROUND_INTENSITY; 
    static const WORD _fg_hi_cyan        = _fg_low_cyan    | FOREGROUND_INTENSITY; 
    static const WORD _fg_hi_magenta     = _fg_low_magenta | FOREGROUND_INTENSITY; 
    static const WORD _fg_hi_yellow      = _fg_low_yellow  | FOREGROUND_INTENSITY;

    static const WORD _bg_black          = 0;
    static const WORD _bg_low_red        = BACKGROUND_RED;
    static const WORD _bg_low_green      = BACKGROUND_GREEN;
    static const WORD _bg_low_blue       = BACKGROUND_BLUE;
    static const WORD _bg_low_cyan       = _bg_low_green   | _bg_low_blue;
    static const WORD _bg_low_magenta    = _bg_low_red     | _bg_low_blue;
    static const WORD _bg_low_yellow     = _bg_low_red     | _bg_low_green;
    static const WORD _bg_low_white      = _bg_low_red     | _bg_low_green | _fg_low_blue;
    static const WORD _bg_gray           = _bg_black       | BACKGROUND_INTENSITY;
    static const WORD _bg_hi_white       = _bg_low_white   | BACKGROUND_INTENSITY;
    static const WORD _bg_hi_blue        = _bg_low_blue    | BACKGROUND_INTENSITY;
    static const WORD _bg_hi_green       = _bg_low_green   | BACKGROUND_INTENSITY;
    static const WORD _bg_hi_red         = _bg_low_red     | BACKGROUND_INTENSITY;
    static const WORD _bg_hi_cyan        = _bg_low_cyan    | BACKGROUND_INTENSITY;
    static const WORD _bg_hi_magenta     = _bg_low_magenta | BACKGROUND_INTENSITY;
    static const WORD _bg_hi_yellow      = _bg_low_yellow  | BACKGROUND_INTENSITY;
    

private:
    HANDLE                      _con_handle;
    DWORD                       _con_size;
    DWORD                       _chars_written; 
    CONSOLE_SCREEN_BUFFER_INFO  _info; 
    CONSOLE_SCREEN_BUFFER_INFO  _def_info; 

public:
    default_console()
      : _con_handle(GetStdHandle(STD_OUTPUT_HANDLE))
    { 
        GetConsoleScreenBufferInfo(_con_handle, &_def_info);
    }
public:
    void
    clear()
    {
        COORD screen_coord = {0, 0};
            
        get_info(); 
        FillConsoleOutputCharacter(_con_handle, ' ', _con_size, screen_coord, &_chars_written); 
        get_info(); 
        FillConsoleOutputAttribute(_con_handle, _info.wAttributes, _con_size, screen_coord, &_chars_written); 
        SetConsoleCursorPosition(_con_handle, screen_coord); 
    }
    void
    set_color(const WORD rgbi, const WORD mask)
    {
        get_info();
        _info.wAttributes &= mask; 
        _info.wAttributes |= rgbi; 
        SetConsoleTextAttribute(_con_handle, _info.wAttributes);
    }
    void
    reset_color()
    {
        SetConsoleTextAttribute(_con_handle, _def_info.wAttributes);
    }
private:
    void
    get_info()
    {
        GetConsoleScreenBufferInfo(_con_handle, &_info);
        _con_size = _info.dwSize.X * _info.dwSize.Y; 
    }
}; // class default_console

static default_console _scm_util_def_console;
    
inline std::ostream& clr        (std::ostream& os) { os.flush(); _scm_util_def_console.clear();       return os; };
inline std::ostream& reset_color(std::ostream& os) { os.flush(); _scm_util_def_console.reset_color(); return os; };

inline std::ostream& fg_red         (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_hi_red,           default_console::_bg_mask); return os; }
inline std::ostream& fg_green       (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_hi_green,         default_console::_bg_mask); return os; }
inline std::ostream& fg_blue        (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_hi_blue,          default_console::_bg_mask); return os; }
inline std::ostream& fg_white       (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_hi_white,         default_console::_bg_mask); return os; }
inline std::ostream& fg_cyan        (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_hi_cyan,          default_console::_bg_mask); return os; }
inline std::ostream& fg_magenta     (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_hi_magenta,       default_console::_bg_mask); return os; }
inline std::ostream& fg_yellow      (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_hi_yellow,        default_console::_bg_mask); return os; }
inline std::ostream& fg_dk_red      (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_low_red,          default_console::_bg_mask); return os; }
inline std::ostream& fg_dk_green    (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_low_green,        default_console::_bg_mask); return os; }
inline std::ostream& fg_dk_blue     (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_low_blue,         default_console::_bg_mask); return os; }
inline std::ostream& fg_dk_white    (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_low_white,        default_console::_bg_mask); return os; }
inline std::ostream& fg_dk_cyan     (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_low_cyan,         default_console::_bg_mask); return os; }
inline std::ostream& fg_dk_magenta  (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_low_magenta,      default_console::_bg_mask); return os; }
inline std::ostream& fg_dk_yellow   (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_low_yellow,       default_console::_bg_mask); return os; }
inline std::ostream& fg_black       (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_black,            default_console::_bg_mask); return os; }
inline std::ostream& fg_gray        (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_gray,             default_console::_bg_mask); return os; }
inline std::ostream& bg_red         (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_hi_red,           default_console::_fg_mask); return os; }
inline std::ostream& bg_green       (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_hi_green,         default_console::_fg_mask); return os; }
inline std::ostream& bg_blue        (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_hi_blue,          default_console::_fg_mask); return os; }
inline std::ostream& bg_white       (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_hi_white,         default_console::_fg_mask); return os; }
inline std::ostream& bg_cyan        (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_hi_cyan,          default_console::_fg_mask); return os; } 
inline std::ostream& bg_magenta     (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_hi_magenta,       default_console::_fg_mask); return os; }
inline std::ostream& bg_yellow      (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_hi_yellow,        default_console::_fg_mask); return os; }
inline std::ostream& bg_dk_red      (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_low_red,          default_console::_fg_mask); return os; }
inline std::ostream& bg_dk_green    (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_low_green,        default_console::_fg_mask); return os; }
inline std::ostream& bg_dk_blue     (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_low_blue,         default_console::_fg_mask); return os; }
inline std::ostream& bg_dk_white    (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_low_white,        default_console::_fg_mask); return os; }
inline std::ostream& bg_dk_cyan     (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_low_cyan,         default_console::_fg_mask); return os; } 
inline std::ostream& bg_dk_magenta  (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_low_magenta,      default_console::_fg_mask); return os; }
inline std::ostream& bg_dk_yellow   (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_low_yellow,       default_console::_fg_mask); return os; }
inline std::ostream& bg_black       (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_black,            default_console::_fg_mask); return os; }
inline std::ostream& bg_gray        (std::ostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_gray,             default_console::_fg_mask); return os; }

inline std::wostream& clr        (std::wostream& os) { os.flush(); _scm_util_def_console.clear();       return os; };
inline std::wostream& reset_color(std::wostream& os) { os.flush(); _scm_util_def_console.reset_color(); return os; };

inline std::wostream& fg_red         (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_hi_red,           default_console::_bg_mask); return os; }
inline std::wostream& fg_green       (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_hi_green,         default_console::_bg_mask); return os; }
inline std::wostream& fg_blue        (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_hi_blue,          default_console::_bg_mask); return os; }
inline std::wostream& fg_white       (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_hi_white,         default_console::_bg_mask); return os; }
inline std::wostream& fg_cyan        (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_hi_cyan,          default_console::_bg_mask); return os; }
inline std::wostream& fg_magenta     (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_hi_magenta,       default_console::_bg_mask); return os; }
inline std::wostream& fg_yellow      (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_hi_yellow,        default_console::_bg_mask); return os; }
inline std::wostream& fg_dk_red      (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_low_red,          default_console::_bg_mask); return os; }
inline std::wostream& fg_dk_green    (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_low_green,        default_console::_bg_mask); return os; }
inline std::wostream& fg_dk_blue     (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_low_blue,         default_console::_bg_mask); return os; }
inline std::wostream& fg_dk_white    (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_low_white,        default_console::_bg_mask); return os; }
inline std::wostream& fg_dk_cyan     (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_low_cyan,         default_console::_bg_mask); return os; }
inline std::wostream& fg_dk_magenta  (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_low_magenta,      default_console::_bg_mask); return os; }
inline std::wostream& fg_dk_yellow   (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_low_yellow,       default_console::_bg_mask); return os; }
inline std::wostream& fg_black       (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_fg_black,            default_console::_bg_mask); return os; }
inline std::wostream& fg_gray        (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_gray,             default_console::_bg_mask); return os; }
inline std::wostream& bg_red         (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_hi_red,           default_console::_fg_mask); return os; }
inline std::wostream& bg_green       (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_hi_green,         default_console::_fg_mask); return os; }
inline std::wostream& bg_blue        (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_hi_blue,          default_console::_fg_mask); return os; }
inline std::wostream& bg_white       (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_hi_white,         default_console::_fg_mask); return os; }
inline std::wostream& bg_cyan        (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_hi_cyan,          default_console::_fg_mask); return os; } 
inline std::wostream& bg_magenta     (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_hi_magenta,       default_console::_fg_mask); return os; }
inline std::wostream& bg_yellow      (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_hi_yellow,        default_console::_fg_mask); return os; }
inline std::wostream& bg_dk_red      (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_low_red,          default_console::_fg_mask); return os; }
inline std::wostream& bg_dk_green    (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_low_green,        default_console::_fg_mask); return os; }
inline std::wostream& bg_dk_blue     (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_low_blue,         default_console::_fg_mask); return os; }
inline std::wostream& bg_dk_white    (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_low_white,        default_console::_fg_mask); return os; }
inline std::wostream& bg_dk_cyan     (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_low_cyan,         default_console::_fg_mask); return os; } 
inline std::wostream& bg_dk_magenta  (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_low_magenta,      default_console::_fg_mask); return os; }
inline std::wostream& bg_dk_yellow   (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_low_yellow,       default_console::_fg_mask); return os; }
inline std::wostream& bg_black       (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_black,            default_console::_fg_mask); return os; }
inline std::wostream& bg_gray        (std::wostream& os) { os.flush(); _scm_util_def_console.set_color(default_console::_bg_gray,             default_console::_fg_mask); return os; }

#else // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
inline std::ostream& clr        (std::ostream& os) { return os; };
inline std::ostream& reset_color(std::ostream& os) { return os; };

inline std::ostream& fg_red         (std::ostream& os) { return os; }
inline std::ostream& fg_green       (std::ostream& os) { return os; }
inline std::ostream& fg_blue        (std::ostream& os) { return os; }
inline std::ostream& fg_white       (std::ostream& os) { return os; }
inline std::ostream& fg_cyan        (std::ostream& os) { return os; }
inline std::ostream& fg_magenta     (std::ostream& os) { return os; }
inline std::ostream& fg_yellow      (std::ostream& os) { return os; }
inline std::ostream& fg_dk_red      (std::ostream& os) { return os; }
inline std::ostream& fg_dk_green    (std::ostream& os) { return os; }
inline std::ostream& fg_dk_blue     (std::ostream& os) { return os; }
inline std::ostream& fg_dk_white    (std::ostream& os) { return os; }
inline std::ostream& fg_dk_cyan     (std::ostream& os) { return os; }
inline std::ostream& fg_dk_magenta  (std::ostream& os) { return os; }
inline std::ostream& fg_dk_yellow   (std::ostream& os) { return os; }
inline std::ostream& fg_black       (std::ostream& os) { return os; }
inline std::ostream& fg_gray        (std::ostream& os) { return os; }
inline std::ostream& bg_red         (std::ostream& os) { return os; }
inline std::ostream& bg_green       (std::ostream& os) { return os; }
inline std::ostream& bg_blue        (std::ostream& os) { return os; }
inline std::ostream& bg_white       (std::ostream& os) { return os; }
inline std::ostream& bg_cyan        (std::ostream& os) { return os; } 
inline std::ostream& bg_magenta     (std::ostream& os) { return os; }
inline std::ostream& bg_yellow      (std::ostream& os) { return os; }
inline std::ostream& bg_dk_red      (std::ostream& os) { return os; }
inline std::ostream& bg_dk_green    (std::ostream& os) { return os; }
inline std::ostream& bg_dk_blue     (std::ostream& os) { return os; }
inline std::ostream& bg_dk_white    (std::ostream& os) { return os; }
inline std::ostream& bg_dk_cyan     (std::ostream& os) { return os; } 
inline std::ostream& bg_dk_magenta  (std::ostream& os) { return os; }
inline std::ostream& bg_dk_yellow   (std::ostream& os) { return os; }
inline std::ostream& bg_black       (std::ostream& os) { return os; }
inline std::ostream& bg_gray        (std::ostream& os) { return os; }

inline std::wostream& clr        (std::wostream& os) { return os; };
inline std::wostream& reset_color(std::wostream& os) { return os; };

inline std::wostream& fg_red         (std::wostream& os) { return os; }
inline std::wostream& fg_green       (std::wostream& os) { return os; }
inline std::wostream& fg_blue        (std::wostream& os) { return os; }
inline std::wostream& fg_white       (std::wostream& os) { return os; }
inline std::wostream& fg_cyan        (std::wostream& os) { return os; }
inline std::wostream& fg_magenta     (std::wostream& os) { return os; }
inline std::wostream& fg_yellow      (std::wostream& os) { return os; }
inline std::wostream& fg_dk_red      (std::wostream& os) { return os; }
inline std::wostream& fg_dk_green    (std::wostream& os) { return os; }
inline std::wostream& fg_dk_blue     (std::wostream& os) { return os; }
inline std::wostream& fg_dk_white    (std::wostream& os) { return os; }
inline std::wostream& fg_dk_cyan     (std::wostream& os) { return os; }
inline std::wostream& fg_dk_magenta  (std::wostream& os) { return os; }
inline std::wostream& fg_dk_yellow   (std::wostream& os) { return os; }
inline std::wostream& fg_black       (std::wostream& os) { return os; }
inline std::wostream& fg_gray        (std::wostream& os) { return os; }
inline std::wostream& bg_red         (std::wostream& os) { return os; }
inline std::wostream& bg_green       (std::wostream& os) { return os; }
inline std::wostream& bg_blue        (std::wostream& os) { return os; }
inline std::wostream& bg_white       (std::wostream& os) { return os; }
inline std::wostream& bg_cyan        (std::wostream& os) { return os; } 
inline std::wostream& bg_magenta     (std::wostream& os) { return os; }
inline std::wostream& bg_yellow      (std::wostream& os) { return os; }
inline std::wostream& bg_dk_red      (std::wostream& os) { return os; }
inline std::wostream& bg_dk_green    (std::wostream& os) { return os; }
inline std::wostream& bg_dk_blue     (std::wostream& os) { return os; }
inline std::wostream& bg_dk_white    (std::wostream& os) { return os; }
inline std::wostream& bg_dk_cyan     (std::wostream& os) { return os; } 
inline std::wostream& bg_dk_magenta  (std::wostream& os) { return os; }
inline std::wostream& bg_dk_yellow   (std::wostream& os) { return os; }
inline std::wostream& bg_black       (std::wostream& os) { return os; }
inline std::wostream& bg_gray        (std::wostream& os) { return os; }

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

} // namespace util
} // namespace scm

#endif // SCM_CORE_LOG_CONSOLE_COLOR_H_INCLUDED

