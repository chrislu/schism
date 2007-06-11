/* $Id: vv_shm.h,v 1.1.1.1 2001/10/11 12:51:34 johnny Exp $ */
/***************************************************************************
          Copyright (c) 1988 - 1995  Vital Images, Inc.
                        1996 - 1997  CogniSeis Development
                           All Rights Reserved


        $Source: /home/cvsadmin/octreemizer/octreemizer/include/libOctreemizer/vv_shm.h,v $
        $Revision: 1.1.1.1 $
        $Date: 2001/10/11 12:51:34 $
        $State: Exp $
        $Locker:  $

***************************************************************************/
#ifdef __IDENT__
static char *VV_SHM_H =
 "$Header: /home/cvsadmin/octreemizer/octreemizer/include/libOctreemizer/vv_shm.h,v 1.1.1.1 2001/10/11 12:51:34 johnny Exp $";
#endif

#ifndef __VV_SHM__
#define __VV_SHM__


#ifdef __cplusplus
extern "C" {
#endif

#ifdef sun
void sginap(long);
#endif

/****************** VoxelView unit definitions ******************/
/**** Must be identical to utility/conversions.h definitions ****/
/****************************************************************/

#ifndef VG_Units
/*
 * This enum defines units flags.  It is needed for units conversions.
 * These odd values (except VG_NONE and VG_UNKNOWN) come from the GeoView
 * toolkit, so we can directly use them for imports/exports to other
 * CogniSies programs.
 */
enum VG_Units {
    VG_METERS                = 0,
    VG_DECIMETERS            = 1,
    VG_CENTIMETERS           = 2,
    VG_KILOMETERS            = 3,
    VG_FEET                  = 10,
    VG_DECIFEET              = 11,
    VG_MILES                 = 13,
    VG_SECONDS               = 30,
    VG_MILLISECONDS          = 35,
    VG_MICROSECONDS          = 37,
    VG_FEET_PER_SECOND       = 40,
    VG_METERS_PER_SECOND     = 50,
    VG_KILOMETERS_PER_SECOND = 53,
    VG_NONE                  = 254,
    VG_UNKNOWN               = 255
};
#endif

/**** VoxelView shared memory constant definitions ****/
/******************************************************/

#define SEG_NAME_MAX 50
#define SHMSEG_MAX  100    /* see SHMMNI in kernel master files */


/*
 * Shared memory data types.
 * See SegBit enum below before changing.
 */
typedef int VV_ShmIndex;
typedef enum VV_SharedNum {
    VV_SHARED_TYPE = 0,
    VV_VOLUME_TYPE,
    VV_RENDER_TYPE,
    VV_GEO_TYPE,
    VV_PICTURE_TYPE,
    VV_MEASURE_TYPE,
    VV_HISTOGRAM_TYPE,
    VV_MAX_SHM_TYPE        /* MUST be last item in this enum */
} VV_ShmType;

/*
 * This enum must be kept in sync with the VV_SharedNum enum such that the
 * relationship of 2^VV_SharedNum == SegBit for each VV_SharedNum remains
 * true.  Otherwise the inner mechanics of vv_getShm() will not work.
 */
enum SegBit {
    VV_SHARED_SEG  =  1,
    VV_VOLUME_SEG  =  2,
    VV_RENDER_SEG  =  4,
    VV_GEO_SEG     =  8,
    VV_PICTURE_SEG = 16,
    VV_MEASURE_SEG = 32
};

struct shmDataVals {
    char volName[SEG_NAME_MAX+1];

    int shrIndex;
    struct sh_var *shrPtr;

    int volIndex;
    struct VV_volume *volPtr;

    int renIndex;
    struct VV_render *renPtr;

    int picIndex;
    struct VV_picture *picPtr;

};

/*
 * NOTE: type_names and environmentNames are becoming obselete, but
 * as long as it survives,  type_names must still be kept in sync with
 * the typeNames array in vv_shm.c, and all must correspond exactly to
 * VV_SharedNum above.
 */
/*
static char *type_names[] = {"SH_VAR", "VOLUME", "RENDER", "GEOMETRY",
                             "PICTURE", "MEASURE", "HISTOGRAM"};
*/

/* used with getenv() to retrieve segment index */
/*
 * NOTE: This is no longer neccesary since vv_getEntryInfoByName and
 * vv_getEntryInfoNameByNumber return only indexes that belong to their
 * process group.  New coding should avoid the use of this array, as
 * it will become obselete.
 */
/*
static char *environmentNames[] =
{
    "VV_SHARE_INDEX", "VV_VOLUME_INDEX", "VV_RENDER_INDEX", "VV_GEO_INDEX",
    "VV_PICTURE_INDEX", "VV_MEASURE_INDEX"
};
*/


/* Loading status flag values (see VV_volume.flag) */
/* ----------------------------------------------- */
#define VV_LOADING         2     /* volume is still being loaded */
#define VV_LOADED          1     /* volume is now resident in shared memory */
#define VV_EXIT_MAIN_PROG  0     /* VoxelLoader was exited during loading */
#define VV_LOAD_CANCEL     3     /* VoxelLoader was canceled during loading */
#define VV_NOT_LOADED     -1     /* volume loading was unsuccessful/cancelled */
#define VV_ATTACH_FAILED  -100   /* attachment to VOLUME segment failed */


/*   Definitions for the shared memory create, attach,
 *    detach & clone functions flags
 *   NOTE: these values MUST remain negative!!! (see vv_clean_shm() in init.c))
 * ---------------------------------------------------- */
typedef enum {VV_REMOVE_USER      = -1, /* Remove segment */
              VV_REMOVE_AND_STAY  = -2  /* Retain segment in memory */
             } vv_DetachVal;


/* The following define is a special value which may be found in
 * VV_volume.segCreatorProcGrp for a given VV_volume segment.  It indicates
 * that vv_detach_shm() was called for a volume segment with the flag
 * VV_REMOVE_AND_STAY.  This means that the shared memory segment
 * will remain in memory, even with no processes attached.  This
 * allows starting VoxelView on that volume without the time-consuming
 * reloading of the data.
 */
#define VV_LEAVE_RESIDENT 0



/* Volume forms */
/* ------------ */ /* (see VV_volume.volume_form below) */
#define VV_NO_VOL               0   /* no volume data is resident in shared memory */
#define VV_FULL_VOL             1   /* only verbatim (unencoded) data is resident */
#define VV_ENC_VOL              2   /* voxel-encoded data structure is also resident */
#define VV_ENCGRD_VOL           4   /* voxel-gradient-encoded structure is resident (5 bytes/voxel) */
#define VV_VOXGMAG_VOL          8   /* voxel-gradient magnitude 16 bit structure is resident */
#define VV_VOXNRM_VOL          16   /* voxel-normal 16 bit structure is resident */
#define VV_ENCFAST_VOL         32   /* voxel-fast-encoded structure is resident (3 bytes/voxel) */


/* Volume tag fields */
/* ----------------- */ /* (see VV_volume.tags below) */
#define VV_CLEAR_VOL_TAGS         0   /* for initialization only */
#define VV_PARTIAL_VOL_TAG        1   /* subvolume has been saved as a volume file */
#define VV_CATEGORIES             4   /* Categories are been used in current volume (data lsb corrupted) */
#define VV_MULTI_LOAD             8   /* Current data set has been loaded a mutiple sister volumes for multichannel display */
#define VV_MULTI_LOAD_LAST       16   /* Last volume in a multi-load operation */
#define VV_DIR_SCALE             32   /* This volume requires direct scaling */
#define VV_RESAMPLED             64   /* This volume is resampled */
#define VV_ENABLE_FAST_LIGHTING 128   /* This volume needs fast lighting */
#define VV_MULTIRESOLUTION      256   /* This is a multi-resolution volume */
#define VV_MODIFIED_VOL         512   /* Loader edited volume or slice files */

/* Volume header magic numbers */
/* --------------------------- */ /* (see VV_volume.magic below) */
#define VV_VOL_VERSION_2_0         0xABC2   /* version 2.0 header, unique magic number */
#define VV_VOL_CURRENT_VERSION     VV_VOL_VERSION_2_0

/* Render segment control flag values */
/* ---------------------------------- */ /* (see VV_render.coloron below) */
#define    VV_NOCOLOR            0
#define    VV_VOXCOLOR           1
#define    VV_GRADCOLOR          2
/* ---------------------------------- */ /* (see VV_render.lighting below) */
#define    VV_NO_LIGHTING        0
#define    VV_PREC_LIGHTING      1
#define    VV_FAST_LIGHTING      2
/* ---------------------------------- */ /* (see VV_render.antialias below) */
#define    VV_NO_ANTIALIAS       0
#define    VV_PREC_ANTIALIAS     1
#define    VV_FAST_ANTIALIAS     2
/* ---------------------------------- */ /* (see VV_render.light_reference below) */
#define    VV_BOUND_TO_VIEWER    0
#define    VV_BOUND_TO_VOLUME    1
/* ---------------------------------- */ /* (see VV_render.gradient_method below) */
#define    VV_SIX_NEIGHBORS      0
#define    VV_TWELVE_NEIGHBORS   1
/* ---------------------------------- */ /* (see VV_render.render_mode below) */
#define    VV_NOVOLUME           0
#define    VV_AUTO               1
#define    VV_MANUAL             2
#define    VV_BOX                4
#define    VV_SLICE_3D           8
#define    VV_SLICE_2D          16
/* ---------------------------------- */ /* (see VV_render.render_direction below) */
#define    VV_X                  0
#define    VV_Y                  1
#define    VV_Z                  2
/* ---------------------------------- */ /* (see VV_render.geomety_state below) */
#define    VV_GEOMETRY_OFF       0
#define    VV_GEOMETRY_FULL      1
#define    VV_GEOMETRY_VOLUME    2
#define    VV_GEOMETRY_SLICE     4
#define    VV_GEOMETRY_SUBVOLUME 8

/* Display modes */
/* ------------- */ /* (see VV_window below) */
#define VV_DOUBLEBUFFER           1 /* display is double-buffered */
#define VV_LEVELS               256 /* Current number of voxel values, i.e. 2^8 */
#define VV_LEVELS_16          65536 /* Alternate number of voxel values, i.e. 2^16 */


    /**** VoxelView shared variable type definitions ****/
    /****************************************************/


typedef enum AreaType {
    SUBVOLUME = 0,
    VOLUME,
    STUDY_AREA
    } AreaType;

typedef enum BoxType {
    BoxType_BOX = 0,
    BoxType_AXES,
    BoxType_NONE
    } BoxType;

typedef enum AnnotationType {
    SLICE_NUMBER = 0,
    SURVEY_COORDINATES,
    WORLD_COORDINATES
    } AnnotationType;

/******** ANNOTATION STRUCTURES **********/

struct vvAnnotations {
    int       on_off;                  /* annotation on/off flag                              */
    AnnotationType   coord;            /* "SLICE", "SURVEY" or "WORLD"                        */
    BoxType   boxType;                 /* "BOX", "AXES" or "NONE"                             */
    int       lineFlag;                /* Boundary line on/off flag                           */
    int       ticMarkFlag;             /* Tic mark flag (0-none, 1-on axes, 2-outside volume) */
    int       ticLblFlag;              /* Tic mark label on/off flag                          */
    int       axisLblFlag;             /* Axis label on/off flag                              */
    int       northArrowFlag;          /* North arrow on/off flag                             */
    float     axisThickness;           /* Thickness of axes                                   */
    float     boxThickness;            /* Thickness of bounding box lines                     */
    float     ticSpacing[3];           /* Tic mark spacing (small tics)                       */
    int       ticsPerLabel[3];         /* # of small tics per labelled (large) tic            */
    char      ticFontName[64];         /* Full name of X font used to diaplay tic labels      */
    char      axisFontName[64];        /* Full name of X font used to display axis labels     */
    int       ticFontSize;             /* Size (in points) of tic label font                  */
    char      ticFontStyle[16];        /* Tic label font style (e.g., "Courier")              */
    int       axisFontSize;            /* Size (in points) of axis label font                 */
    char      axisFontStyle[16];       /* Axis label font style (e.g., "Courier")             */
    float     axisColor[3];            /* Components of axis color                            */
    float     boxColor[3];             /* Components of box color                             */
    float     ticMarkColor[3];         /* Components of tic mark color                        */
    float     ticLblColor[3];          /* Components of tic mark label color                  */
    float     axisLblColor[3];         /* Components of axis label color                      */
};


/******** END OF ANNOTATION STRUCTURES **********/

/********     SUBVOLUME STRUCTURES    **********/

struct vvSubvolume {
    int seedPoint[3];                  /* pick location voxel value                  */
    int connectivity;                  /* 0,6,26 connectivity values                 */
    int amplitude_range;               /* Amplitude range checked                    */
    float rangeValues[2];              /* Amplitude range values                     */
    int amplitude_gradient;            /* Amplitude gradient checked                 */

    float gradient_values[2];          /* Amplitude gradient values                  */
    int hex_compare;                   /* Hex compare checked                        */
    int hex_OrAnd_switchs[2];          /* Hex compare or and checks                  */
    float hex_values[2];               /* Hex compare values                         */
    int embedded_geometry_limit;       /* Embedded Geometry Limit Checked            */
};

/******** END OF SUBVOLUME STRUCTURES **********/

struct VV_picture {
    int  flag;              /* flag for passing messages */
    int  redraw_count;      /* serial count to indicate redraw of picture */
    int  saved;             /* flag indicating the picture has been saved */
    int  depth;                      /* bits per pixel */
    int  xsize, ysize;               /* size in pixels */
    int  xorigin, yorigin;           /* origin of window within display (from lower left) */
    int  numpics;                    /* number of images */
    int  windepth;                   /* rendering window depth in the screen */
    int  pop_count;                  /* serial count to indicate the need to pop the window */
    int  stow_count;                 /* serial count to indicate the need to close the window
                                        because the host program is stowed */
    char segName[SEG_NAME_MAX];      /* Segment name, from data set dir name */
    int segCreatorProcGrp;           /* Process group of creator process */
    int  last_xsize, last_ysize;     /* size in pixels during last call to get_window_contents */
};

struct VV_volume {
    int magic;                                 /* magic number for binary file id */
    int volume_form;                           /* volume type flag (see 'Volume forms' above) */
    unsigned int size;                         /* volume segment size (in bytes) */
    int flag;                                  /* loading status flag (see 'Loading status flag values' above) */
    unsigned count;                            /* serial number to indicate updates */
    char pathname[300];                        /* pathname of volume data set */
    int databits;                              /* # of data bits per voxel */
    int normbits;                              /* # of normal bits per voxel */
    int gradbits;                              /* # of gradient bits per voxel */
    int voxelbits;                             /* total # of bits per voxel */
    int xsize, ysize, zsize;                   /* volume dimensions (pixels) */
    int interspace;                            /* interpolation factor for slices */
    float voffset, xoffset, yoffset, zoffset;  /* world coord offset */
    float vcal, xcal, ycal, zcal;              /* calibration factors (units/voxel) */
    char vunit[16];                            /* physical units for voxels */
    char xunit[16], yunit[16], zunit[16];      /* physical units for coord's */
    char vlabel[16];                           /* description for voxels */
    char xlabel[16], ylabel[16], zlabel[16];   /* description for coord's */
    unsigned int  histogram[VV_LEVELS];        /* voxel histogram */
    unsigned int  grad_histogram[VV_LEVELS];   /* gradient histogram */

    int tags;                                  /* 32 tag bits for special indicators */
    unsigned voxels_changed_count;             /* message flag for indicating that some voxels have changed:
                                                * incremented by marking algorithms and read by rendering engine
                                                * to determine if caches need updating. */

    /* Always add the values immediately after
       VV_reserved space to keep other variables
       at the same offset as before.
    */
    char VV_reserved[226-160-SEG_NAME_MAX];    /* reserved for VoxelGeo */
                                               /* Taken from VV_reserved:
                                                *  4 for volSurveyUnits
                                                *  4 for volWorldUnits
                                                * 96 for worldXref
                                                *  4 for dummy
                                                *  4 for worldFlag
                                                * 48 VV_reserved2 */
    int volSurveyUnits;                        /* Volume units flags
(survey coordinates): 0xAAxxyyzz, where:
                                                *     xx = x-dimension units flag
                                                *     yy = y-dimension units flag
                                                *     zz = z-dimension units flag
                                                *  See VG_Units above for units flag definitions */
    int volWorldUnits;                         /* Volume units flags (world coordinates): 0xAAxxyyzz, where:
                                                *     xx = x-dimension units flag
                                                *     yy = y-dimension units flag
                                                *     zz = z-dimension units flag
                                                *  See VG_Units above for units flag definitions */
    double worldXref[4][3];                    /* Three reference points (from _world file):
                                                *  worldXref[0][i] = World X (i = point #)
                                                *  worldXref[1][i] = World Y (i = point #)
                                                *  worldXref[2][i] = Survey Y (i = point #)
                                                *  worldXref[3][i] = Survey X (i = point #) */
    int dummy;                                 /* Unused variable to reset alignment */
    int worldFlag;                             /* flag to specify if the
                                                * _world file is present */
    float VV_reserved2[12];                    /* reserved for VoxelGeo */

    int orig_x, orig_y, orig_z;                /* original volume dimensions */
    int catbits;                               /* # of category bits per voxel */

    short spacer;                              /* make sure segName starts on the same byte as before */
    char segName[SEG_NAME_MAX];                /* Segment name, from data set dir name */
    int segCreatorProcGrp;                     /* Process group of creator process */

    char user_space[256];
};

struct VV_render {
  unsigned int  general_count;        /* message flag for any segment changes */
  unsigned int  rerender_count;       /* message flag for triggering re-rendering */
  unsigned int  abort_count;          /* message flag for interrupting rendering */

  int     low_voxel, high_voxel;               /* voxel threshold values [0, VV_LEVELS] */
  int     offset;                              /* voxel transform DC offset [0, VV_LEVELS] */
  float   gain;                                /* voxel transform gain [0.0, 256.0] */
  int     grad_threshold;                      /* gradient threshold */

  int     trim_volume;                         /* trim volume mode flag */
  int     geomety_state;                       /* geometry states */
  int     fast_embedding;                      /* fast embedding */

  int     xmin, ymin, zmin;                    /* minimum orthogonal limtis */
  int     xmax, ymax, zmax;                    /* maximum orthogonal limtis */
  int     chair_xpos,chair_ypos,chair_zpos;    /* Chair cut poistions */
  int     current_position[3];                 /* current render positions */
  float   scale_x, scale_y, scale_z;           /* Volume Scales */
  int     slab_x, slab_y,slab_z;               /* Slab mode thickness */
  int     render_mode;                         /* render mode */
  int     render_direction;                    /* render direction */
  int     scrim;                               /* scrim on/off flag */

  int     lighting, coloron, antialias;        /* still more flags */


/* Lighting  Parameters */

  float   contrast;                            /* image contrast control */
  float   light_ambient_coeff[4],
          light_direction[4],
          light_diffuse_coeff[4],
          light_specular_coeff[4];             /* light properties */
  float   shininess;                           /* voxel material property */
  int     objectLighting;                      /* set if lighting is turned on or off */

/* Static Parameters */

  int     totalvoxels;                         /* total number of voxels currently rendered */
  int     totalintegral;                       /* voxels currently rendered
                                                * voxel value */
  unsigned int subset_histogram[VV_LEVELS];   /* subset voxel histogram */
  unsigned int subset_grad_histogram[VV_LEVELS]; /* subset grad. histogram */


/* Lookup Table Parameters */

  char     hue_filename[256];                  /* name of hue file to use */
  char     color_palette[256];                 /* color pallette name */
  char     opacity_table[256];                 /* opacity table name */
  char     sets_filepath[1024];                 /* settings file name */
  int      fullOpaque;                         /* set engine completely opaque */

  int      threshold_lut[VV_LEVELS];           /* table to turn voxel values on/off [1/0] */
  float    hues_lut[VV_LEVELS][3];             /* voxel value color look-up table [R,G,B] */
  int      alpha_lut[VV_LEVELS];               /* opacity value look-up table */
  int      voxel_lut[VV_LEVELS];               /* 8-bit data to 8-bit voxel value mapping */


/* Annotation Parameters */
  struct   vvAnnotations volume_annot;         /* New (VG2) annotations for the volume */
  struct   vvAnnotations subvol_annot;         /* New (VG2) annotations for the subvolume */
  struct   vvAnnotations area_annot;           /* New (VG2) annotations for the subvolume */

/* View Parameters */

  int      couple;                             /* mode flags for coupling views */
  int      stereo;                             /* mode flags */
  float    stereo_angle;                       /* additional transform angle for stereo */
  int      camera_type;                        /* perspective- 1, orthographic-0 */
  float    camera_position[3];                 /* VG2 camera position */
  float    camera_orientation[4];              /* VG2 camera orientation */
  float    camera_focalDistance;               /* VG2 camera focal distance */
  float    camera_height;                      /* VG2 camera height */
  float    camera_heightAngle;                 /* VG2 camera height angle */
  float    camera_far_distance,
           camera_near_distance;               /* camera near & far distance   */
  int      slice_flip[3];                      /* orientation for 2d slice mode */
  float    slice_rotation[3];                  /* rotation for 2d slice mode 0-4 */


/* Other Parameters */
  int      cursor3d;                           /* use a 3D cursor instead of faces */
  int      is3dcursorActive;                   /* is 3D cursor active */
  int      image_xsize;                        /* x size of image window */
  int      image_ysize;                        /* y size of image window */
  int      picture_segment;                    /* picture shm segment flag */

  int      cursorXRef;                         /* mode flag for cursor cross ref */

  struct vvSubvolume seedCriteria;             /* subvolume window criteria */
  int      embedded_cursor_on;                 /* determines if cursor is on */
  float    embedded_cursor[3];                 /* location of embedded cursor */
  float    background_rgb[3];                  /* background RGB values  */
  int      states;                             /* Indicators for valid rendering options */

  float    xrot, yrot, zrot;                   /* rotational position (tenths) [0.0, 360.0] */
  int      init_lut_count;                     /* Used to trigger LUT initialization */

  char     segName[SEG_NAME_MAX];              /* Segment name, from data set dir name */
  int      segCreatorProcGrp;                  /* Process group of creator process */
  int      segLock;                            /* stores the pid of locking program */
  int      caching[3];                         /* flags to set render caching on/off per direction */

  char     VV_reserved[1024];                  /* for future use by Vital Images */
  char     user_space[256];                    /* for user customization */
};


/*
 * header offset macro:
 * Returns a pointer to the first memory location after a given data structure.
 * Used for getting a pointer to the beginning of the data appended to a data
 * structure. For example, to get a pointer to the volume data
 * appended to the VV_volume structure.
 */
#define vv_shared_data(ptr,type)    ((unsigned char *)(ptr) + sizeof(struct type))

/*
 * error defs for vv_clone_shm()
 */
#define CLONE_SEG_CREATE_FAILED -1 /* unsuccessful cloning - new segment was not created. */
#define CLONE_SEG_ATTACH_FAILED -2 /* unsuccessful cloning - new segment attached failed. */
#define CLONE_IPC_STAT_FAILED   -3 /* couldn't get the size of the original segment. */
#define CLONE_NAME_UPDT_FAILED  -4 /* couldn't update name in new segment. */
#define CLONE_CPGP_UPDT_FAILED  -5 /* couldn't update creator process group. */
#define CLONE_NAME_READ_FAILED  -6 /* couldn't read name in new segment. */
#define CLONE_CPGP_READ_FAILED  -7 /* couldn't read creator process group. */

/*
 * vv_ipcsInfo
 * Structure and function to obtain current Vital Images shm seg info.
 */
typedef struct {
    char segName[SEG_NAME_MAX];   /* Segment name, from data set dir name */
    int segCreatorProcGrp;        /* Process group of creator process */
    int shmID;                    /* IPC shared memory ID */
    int shmKey;                   /* IPC key */
    VV_ShmType Type;              /* Member of VV_SharedNum enum */
    VV_ShmIndex Index;            /* i'th segment of this type */
    int nAttached;                /* Number of processes attached to seg */
} vv_singleSeg;

struct vv_segInfo                 /* used by vv_ipcsInfo */
{
    int nSegs;                    /* Number of Vital Images shared memory segments */
    vv_singleSeg segList[SHMSEG_MAX];
};

/*
 *
 *  FUNCTION PROTOTYPES:
 *  These functions are prototyped in the same order as
 *  the documentation. See chapter on " Programming Shared
 *  Memory Interface" for a detailed description of their
 *  usage.
 *
 */
int vv_create_shm(int type, int  size, const char *name);
void *vv_attach_shm(int type,int index);
int vv_detach_shm(int type, int index, void *shared_mem_ptr, int remove_flag);
void *vv_clone_shm(int type, int size, const char *name, void *shared_mem_ptr, int orig_index, int *new_index);
struct vv_segInfo *vv_ipcsInfo(void);
void *vv_cpVolumeHeader(struct VV_volume *tgtVol, const struct VV_volume *srcVol);
int vv_getEntryInfoByName(const char *data_name, const char *type_name,int *shmid, int *attachedCount);
int vv_sem_lock(int sem_num);
int vv_sem_unlock(int sem_num);
int vv_sem_is_locked_by_me(int sem_num);
int vv_clear_sems(void);
int vv_rm_sem(void);
int vv_init_sem(void);
int vv_get_shm_cpid(char *type, int index);

/*
 * UNDOCUMENTED FUNCTION PROTOTYPES:
 * This functions are restricted to internal VoxelView usage at the
 * moment.
 */
int vv_segIndexByName(const char *data_name, const char *type_name);
int vv_getEntryNameByNumber(char *data_name, const char *type_name, int *shmid,int *nAttached, int index);
char *vv_segNameByIndex(const char *type_name, int index);
char *typeName(const VV_ShmType shmType);
int vv_getShmV(int argCount, char *argVec[], int segFlag, struct shmDataVals *Ptr);
int vv_getShm(char *volName, int segFlag, struct shmDataVals *Ptr);
int vv_freeShm(int segFlag, struct shmDataVals *Ptr);
int vv_isVolName(char *Name);

#ifdef __cplusplus
}
#endif
#endif /* __VV_SHM__ */
