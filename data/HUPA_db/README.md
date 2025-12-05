# ORGANISATION OF THE HUPA DATABASE

## 1. FOLDER STRUCTURE

Root:
```
HUPA_db
├── healthy
│   ├── 50 kHz
│   └── 25 kHz
└── pathological
    ├── 50 kHz
    └── 25 kHz
```

* The folder "healthy" contains recordings from healthy speakers.
* The folder "pathological" contains recordings from speakers with a voice disorder.
* Within each of these, there are two subfolders according to the target sampling rate:

  * "50 kHz"  -> mono signals at 50,000 Hz (new filenames, see Section 3).
  * "25 kHz"  -> mono signals at 25,000 Hz (original filenames).

## 2. AUDIO FORMAT

All audio files are:

* Format: .wav
* Channels: mono
  (recordings that were originally stereo have been converted to mono).
* Bit depth: 16-bit PCM (PCM_16).
* Sampling rate:

  * 50 kHz in the "50 kHz" subfolder.
  * 25 kHz in the "25 kHz" subfolder.

**Important note on filenames and distribution**

* In the **25 kHz** folders, the audio files keep their **original filenames**.  
  This 25 kHz version corresponds to the copy of the corpus that has historically been
  distributed to the different research groups that requested access to HUPA. For that
  reason, the original naming convention has been preserved in this branch of the data.
* In the **50 kHz** folders, the audio files use the **new systematic filenames**
  described in Section 3 (RRR_PATIENTCODE_SEX_AGE_CONDITION.wav).  

The correspondence between the **new 50 kHz filenames** and the **original filenames**
is given in `HUPA_db.xlsx` via the columns `Original file name` and `File name`, respectively in Healthy and Pathological worksheets.

## 3. FILE-NAMING CONVENTION (50 kHz VERSION)

For the 50 kHz version, audio files follow the naming convention:

```
RRR_PATIENTCODE_SEX_AGE_CONDITION.wav
```

where:

* RRR:

  * Row identifier (rowID) with three digits (001, 002, 003, ...).
  * It is unique for each recording within the spreadsheet.
  * It allows unambiguous identification of each recording, even when other fields coincide.

* PATIENTCODE:

  * Numerical code representing the voice pathology, as defined in the "Patient code"
    and "Pathology" fields of the spreadsheet.
  * It summarises the speaker's main diagnosis (e.g. nodules, polyp, sulcus, oedema, etc.).
  * For healthy speakers, the value 0 is used, indicating absence of pathology.

* SEX:

  * Speaker's sex:

    * M -> Male
    * F -> Female

* AGE:

  * Speaker's age in years at the time of recording.

* CONDITION:

  * Global label of the vocal condition:

    * healthy      -> healthy speakers
    * pathological -> speakers with a voice disorder

Examples (50 kHz version):
001_0_M_20_healthy.wav  
045_0_F_22_healthy.wav  
001_113_M_45_pathological.wav  
010_212_F_23_pathological.wav  

In these examples:

* 0 indicates absence of pathology (healthy).
* 113, 212, etc. are pathology codes used in the HUPA_db metadata.

For the **25 kHz version**, the files retain the **original filenames** used in the
first distributed versions of the corpus.

## 4. REFERENCE METADATA SPREADSHEET: HUPA_db.xlsx

The HUPA database is documented and linked to the audio files via the spreadsheet:

```
HUPA_db.xlsx
```

This spreadsheet contains four worksheets:

```
- Intro
- Healthy
- Pathological
- Pathology classification
```

### 4.1 Worksheet "Healthy"

* One row per recording from a healthy speaker.
* Key columns:

  * File name

    * New coded filename used in the 50 kHz version
      (RRR_PATIENTCODE_SEX_AGE_CONDITION.wav).
    * It matches exactly the names of the audio files stored under `healthy/50 kHz`.

  * Original file name

    * Original filename preserved in the 25 kHz version.
    * It matches exactly the audio files stored under `healthy/25 kHz` and corresponds
      to the naming convention used in earlier distributed copies of the corpus.

  * Sampling frequency  (first column)

  * Sampling frequency2  (second column)

    * Two sampling-frequency fields, one per column.
    * They store the sampling rate information for each recording (e.g. for different
      versions of the signal, such as original and resampled).

  * Type

    * Global class label for the recording:

      * healthy

  * EGG

    * Indicates whether an electroglottographic (EGG) signal is available or relevant
      for this case (if applicable).

  * Age

    * Speaker's age in years.

  * Sex

    * Speaker's sex:

      * M -> Male
      * F -> Female

  * G, R, A, S, B

    * Perceptual GRBAS-scale ratings:

      * G -> Grade (overall severity)
      * R -> Roughness
      * A -> Asthenia
      * S -> Strain
      * B -> Breathiness

  * Total

    * Global or combined GRBAS score, according to the original clinical protocol.

  * Patient code

    * Numerical code associated with the pathology group or clinical category.
    * For healthy speakers, this is typically 0.

  * Pathology

    * Textual description of the pathology corresponding to "Patient code".
    * For healthy speakers this typically indicates "healthy" or absence of pathology.

  * F0, F1, F2, F3

    * Fundamental frequency (F0) and the first three formant frequencies (F1–F3),
      measured for the sustained vowel or speech segment, depending on the protocol.

  * Formants

    * Additional formant-related information or summary descriptor.

  * Peaks

    * Information about spectral or temporal peaks (e.g. number or amplitude), or a
      derived acoustic measure.

  * Jitter

    * Acoustic jitter measure, describing cycle-to-cycle F0 perturbation.

  * Comments

    * Free-text field with additional notes about the recording, clinical observations
      or deviations from the standard protocol.

### 4.2 Worksheet "Pathological"

* One row per recording from a speaker with a voice pathology.

* The columns are the same as in the "Healthy" worksheet:

  * File name
  * Original file name
  * Sampling frequency (two columns)
  * Type
  * EGG
  * Age
  * Sex
  * G, R, A, S, B
  * Total
  * Patient code
  * Pathology
  * F0, F1, F2, F3
  * Formants
  * Peaks
  * Jitter
  * Comments

* In this worksheet, "Type" is "pathological", and "Patient code" / "Pathology" specify
  the corresponding pathology for each recording.

### 4.3 Worksheet "Intro"

* Summary worksheet describing the database at a global level.
* Contains descriptive statistics and summaries such as:

  * distribution of age,
  * distribution of sex,
  * distribution of GRBAS scores,
  * and the number of speakers/recordings per group.

It provides an overview of the composition of the HUPA corpus.

### 4.4 Worksheet "Pathology classification"

* Lookup table for the pathology codes used in HUPA_db.

* Contains at least:

  * Code

    * Numerical identifier of the pathology group.

  * Pathology

    * Text label describing the pathology (e.g. nodules, polyp, sulcus, oedema,
      leukoplakia, etc.).

* This worksheet defines the mapping between each "Patient code" and its corresponding
  clinical diagnosis.


## 5. SUMMARY

* The HUPA database is organised by subject type:

  * "healthy"      -> healthy speakers
  * "pathological" -> speakers with a voice disorder

* For each group there are two signal versions:

  * 50 kHz mono  (new systematic filenames)
  * 25 kHz mono  (original filenames preserved for compatibility with earlier distributions)

* File names in the 50 kHz version encode:

  * row identifier (RRR),
  * pathology code (PATIENTCODE, 0 for healthy speakers),
  * sex (M/F),
  * age,
  * global condition (healthy/pathological).

* The 25 kHz version keeps the original filenames that were used in the copies of the
  corpus distributed to different research groups.

* The HUPA_db.xlsx spreadsheet provides the complete metadata and the explicit mapping
  between both naming schemes:

  * `File name`           -> new coded filename (50 kHz)
  * `Original file name`  -> original filename (25 kHz)

  together with:

  * sampling frequencies,
  * clinical labels (Type, Patient code, Pathology),
  * demographic variables (Age, Sex),
  * perceptual GRBAS scores,
  * acoustic descriptors (F0, formants, peaks, jitter),
  * high-level summaries in "Intro",
  * and the mapping from pathology code to pathology name in "Pathology classification".