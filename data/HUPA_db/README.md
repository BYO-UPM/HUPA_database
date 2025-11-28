# ORGANISATION OF THE HUPA DATABASE

## 1. FOLDER STRUCTURE

Root:
```
HUPA_db
├── healthy
│   ├── 50 kHz
│   └── 44.1 kHz
└── pathological
    ├── 50 kHz
    └── 44.1 kHz
```

* The folder "healthy" contains recordings from healthy speakers.
* The folder "pathological" contains recordings from speakers with a voice disorder.
* Within each of these, there are two subfolders according to the target sampling rate:

  * "50 kHz"   -> mono signals at 50,000 Hz.
  * "44.1 kHz" -> mono signals at 44,100 Hz.

## 2. AUDIO FORMAT

All audio files are:

* Format: .wav
* Channels: mono
  (recordings that were originally stereo have been converted to mono).
* Bit depth: 16-bit PCM (PCM_16).
* Sampling rate:

  * 50 kHz in the "50 kHz" subfolder.
  * 44.1 kHz in the "44.1 kHz" subfolder.

File names are identical in the "50 kHz" and "44.1 kHz" folders; only the sampling rate differs.


## 3. FILE-NAMING CONVENTION

Audio files follow the naming convention:

```
RRR_PATIENTCODE_SEX_AGE_CONDITION.wav
```

where:

* RRR:

  * Row identifier (rowID) with three digits (001, 002, 003, ...).
  * It is unique for each recording within the Excel file.
  * It allows unambiguous identification of each recording, even when other fields coincide.

* PATIENTCODE:

  * Numerical code representing the voice pathology, as defined in the "Patient code"
    and "Pathology" fields of the Excel file.
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

Examples:
001_0_M_20_healthy.wav
045_0_F_22_healthy.wav
001_113_M_45_pathological.wav
010_212_F_23_pathological.wav

In these examples:

* 0 indicates absence of pathology (healthy).
* 113, 212, etc. are pathology codes used in the HUPA_db metadata.


## 4. REFERENCE METADATA FILE: HUPA_db.xlsx

The HUPA database is documented and linked to the audio files via the Excel file:

```
HUPA_db.xlsx
```

This workbook contains four sheets:

```
- Intro
- Normals
- Pathological
- Pathology classification
```

### 4.1 Sheet "Normals"

* One row per recording from a healthy speaker.
* Key columns:

  * File name

    * Final .wav file name in the database
      (RRR_PATIENTCODE_SEX_AGE_CONDITION.wav).
    * It matches exactly the names of the audio files stored under "healthy".

  * Sampling frequency  (first column)

  * Sampling frequency  (second column)

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

### 4.2 Sheet "Pathological"

* One row per recording from a speaker with a voice pathology.

* The columns are the same as in the "Normals" sheet:

  * File name
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

* In this sheet, "Type" is "pathological", and "Patient code" / "Pathology" specify
  the corresponding pathology for each recording.

### 4.3 Sheet "Intro"

* Summary sheet describing the database at a global level.
* Contains descriptive statistics and summaries such as:

  * distribution of age,
  * distribution of sex,
  * distribution of GRBAS scores,
  * and the number of speakers/recordings per group.

It provides an overview of the composition of the HUPA corpus.

### 4.4 Sheet "Pathology classification"

* Lookup table for the pathology codes used in HUPA_db.

* Contains at least:

  * Code

    * Numerical identifier of the pathology group.

  * Pathology

    * Text label describing the pathology (e.g. nodules, polyp, sulcus, oedema,
      leukoplakia, etc.).

* This sheet defines the mapping between each "Patient code" and its corresponding
  clinical diagnosis.


## 5. SUMMARY

* The HUPA database is organised by subject type:

  * "healthy"      -> healthy speakers
  * "pathological" -> speakers with a voice disorder

* For each group there are two signal versions:

  * 50 kHz mono
  * 44.1 kHz mono

* File names encode:

  * row identifier (RRR),
  * pathology code (PATIENTCODE, 0 for healthy speakers),
  * sex (M/F),
  * age,
  * global condition (healthy/pathological).

* The Excel file HUPA_db.xlsx provides the complete metadata:

  * file names,
  * sampling frequencies,
  * clinical labels (Type, Patient code, Pathology),
  * demographic variables (Age, Sex),
  * perceptual GRBAS scores,
  * acoustic descriptors (F0, formants, peaks, jitter),
  * high-level summaries in "Intro",
  * and the mapping from pathology code to pathology name in "Pathology classification".
