import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  cancelAutoStory,
  createAutoStory,
  fetchAutoStorySpeakerTranscript,
  fetchAutoStory,
  fetchAutoStoryProjects,
  fetchSettings,
  importAutoStoryProject,
  rediscoverAutoStorylines,
  reopenAutoStoryReview,
  retryAutoStoryDiscovery,
  retryAutoStoryRender,
  retryAutoStoryline,
  reconstructAutoStoryStory,
  renderAutoStory,
  selectAutoStoryline,
  selectAutoStorylines,
  saveAutoStorySpeakerNames,
  uploadVideo,
  validateAutoStoryRecipe,
  validateAutoStoryPlan,
} from '../api'

const TERMINAL_STATUSES = ['completed', 'failed', 'cancelled']

function TransferProgress({ label, percent, onCancel, cancellable = true }) {
  return (
    <div className="auto-story-transfer" role="status" aria-live="polite">
      <div className="auto-story-transfer-heading">
        <span>{label}</span>
        <span>{percent}%</span>
      </div>
      <div
        className="progress-bar-bg"
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
      >
        <div className="progress-bar-fill" style={{ width: `${percent}%` }}></div>
      </div>
      {cancellable && (
        <button type="button" className="btn btn-ghost btn-sm" onClick={onCancel}>
          Cancel transfer
        </button>
      )}
    </div>
  )
}

function getDefaultStorylineSelection(autoStory) {
  const failedIds = new Set(
    (autoStory?.failed_stories || []).map(story => story.storyline_id),
  )
  const availableIds = new Set(
    (autoStory?.storylines || [])
      .filter(storyline => !failedIds.has(storyline.id))
      .map(storyline => storyline.id),
  )
  const savedSelection = (autoStory?.selected_storyline_ids || [])
    .filter(id => availableIds.has(id))
  return savedSelection.length
    ? savedSelection
    : [...availableIds]
}

function AutoStory() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const fileRef = useRef(null)
  const projectImportRef = useRef(null)
  const uploadController = useRef(null)
  const importController = useRef(null)
  const speakerPreviewRef = useRef(null)
  const planTouched = useRef(false)
  const selectionTouched = useRef(false)
  const speakerNamesTouched = useRef(false)
  const displayedDiscoveryRevision = useRef(null)
  const lastJobUpdatedAt = useRef(null)
  const [settings, setSettings] = useState(null)
  const [settingsError, setSettingsError] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploadFilename, setUploadFilename] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [importingProject, setImportingProject] = useState(false)
  const [importProgress, setImportProgress] = useState(0)
  const [importUploadComplete, setImportUploadComplete] = useState(false)
  const [targetDuration, setTargetDuration] = useState(60)
  const [workflowMode, setWorkflowMode] = useState('auto')
  const [minGapSec, setMinGapSec] = useState(30)
  const [diarizationEnabled, setDiarizationEnabled] = useState(false)
  const [diarizationSpeakers, setDiarizationSpeakers] = useState('auto')
  const [whisperModel, setWhisperModel] = useState('small')
  const [provider, setProvider] = useState('gemini')
  const [retryProvider, setRetryProvider] = useState('')
  const [creating, setCreating] = useState(false)
  const [job, setJob] = useState(null)
  const [stories, setStories] = useState([])
  const [legacyRecipe, setLegacyRecipe] = useState(null)
  const [legacyTimingOptions, setLegacyTimingOptions] = useState(null)
  const [validated, setValidated] = useState(false)
  const [legacyValidated, setLegacyValidated] = useState(false)
  const [validating, setValidating] = useState(false)
  const [selectedStorylineIds, setSelectedStorylineIds] = useState([])
  const [speakerNameDraft, setSpeakerNameDraft] = useState({})
  const [speakerMergeDraft, setSpeakerMergeDraft] = useState({})
  const [speakerTranscriptOpen, setSpeakerTranscriptOpen] = useState(false)
  const [speakerTranscriptTurns, setSpeakerTranscriptTurns] = useState([])
  const [speakerTranscriptOffset, setSpeakerTranscriptOffset] = useState(0)
  const [speakerTranscriptTotal, setSpeakerTranscriptTotal] = useState(0)
  const [speakerCorrections, setSpeakerCorrections] = useState([])
  const [speakerTranscriptLoading, setSpeakerTranscriptLoading] = useState(false)
  const [activeSpeakerPreview, setActiveSpeakerPreview] = useState('')
  const [savedProjects, setSavedProjects] = useState([])
  const [savedProjectsTotal, setSavedProjectsTotal] = useState(0)
  const [savedProjectsError, setSavedProjectsError] = useState('')
  const [projectStorageStatus, setProjectStorageStatus] = useState(null)
  const [loadingProjects, setLoadingProjects] = useState(false)
  const [loadingMoreProjects, setLoadingMoreProjects] = useState(false)
  const [projectSearch, setProjectSearch] = useState('')
  const [reviewBeforeRender, setReviewBeforeRender] = useState(false)
  const [selecting, setSelecting] = useState(false)
  const [rediscovering, setRediscovering] = useState(false)
  const [retryingDiscovery, setRetryingDiscovery] = useState(false)
  const [retryingRender, setRetryingRender] = useState(false)
  const [retryingStorylineId, setRetryingStorylineId] = useState('')
  const [reopeningReview, setReopeningReview] = useState(false)
  const [reconstructingStoryId, setReconstructingStoryId] = useState('')
  const [rendering, setRendering] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [error, setError] = useState('')
  const [connectionError, setConnectionError] = useState('')
  const legacyTouched = useRef(false)

  useEffect(() => {
    let active = true
    fetchSettings()
      .then(data => {
        if (!active) return
        setSettings(data)
        setProvider(data.default_ai_provider || 'gemini')
      })
      .catch(err => {
        if (active) setSettingsError(`Could not load Studio settings: ${err.message}`)
      })
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (jobId) return undefined
    let active = true
    setLoadingProjects(true)
    setSavedProjects([])
    setSavedProjectsTotal(0)
    const timer = setTimeout(() => {
      fetchAutoStoryProjects({ query: projectSearch, offset: 0, limit: 20 })
      .then(result => {
        if (active) {
          setSavedProjects(result.projects || [])
          setSavedProjectsTotal(result.total || 0)
          setProjectStorageStatus(result.storage || null)
          setSavedProjectsError('')
        }
      })
      .catch(err => {
        if (active) setSavedProjectsError(err.message)
      })
      .finally(() => {
        if (active) setLoadingProjects(false)
      })
    }, 180)
    return () => {
      active = false
      clearTimeout(timer)
    }
  }, [jobId, projectSearch])

  const loadMoreProjects = async () => {
    if (loadingMoreProjects || savedProjects.length >= savedProjectsTotal) return
    setLoadingMoreProjects(true)
    setSavedProjectsError('')
    try {
      const result = await fetchAutoStoryProjects({
        query: projectSearch,
        offset: savedProjects.length,
        limit: 20,
      })
      setSavedProjects(current => [...current, ...(result.projects || [])])
      setSavedProjectsTotal(result.total || 0)
      setProjectStorageStatus(result.storage || null)
    } catch (err) {
      setSavedProjectsError(err.message)
    } finally {
      setLoadingMoreProjects(false)
    }
  }

  useEffect(() => {
    if (!jobId) return undefined
    let active = true
    let timeout
    let consecutiveFailures = 0
    lastJobUpdatedAt.current = null
    displayedDiscoveryRevision.current = null
    selectionTouched.current = false
    speakerNamesTouched.current = false
    planTouched.current = false
    legacyTouched.current = false
    setJob(null)
    setStories([])
    setLegacyRecipe(null)
    setLegacyTimingOptions(null)
    setValidated(false)
    setLegacyValidated(false)
    setSpeakerNameDraft({})
    setSpeakerTranscriptOpen(false)
    setSpeakerTranscriptTurns([])
    setSpeakerTranscriptOffset(0)
    setSpeakerTranscriptTotal(0)
    setSpeakerCorrections([])

    const poll = async () => {
      try {
        const data = await fetchAutoStory(jobId)
        if (!active) return
        consecutiveFailures = 0
        setConnectionError('')
        if (lastJobUpdatedAt.current !== data.updated_at) {
          lastJobUpdatedAt.current = data.updated_at
          setJob(data)
        }
        const currentDiscoveryRevision = data.auto_story?.discovery_revision ?? 0
        if (
          data.status === 'awaiting_storyline'
          && displayedDiscoveryRevision.current !== currentDiscoveryRevision
        ) {
          displayedDiscoveryRevision.current = currentDiscoveryRevision
          selectionTouched.current = false
          setSelectedStorylineIds(getDefaultStorylineSelection(data.auto_story))
          speakerNamesTouched.current = false
          setSpeakerNameDraft(data.auto_story?.speaker_names || {})
          setSpeakerMergeDraft(data.auto_story?.speaker_merges || {})
        } else if (
          data.status === 'awaiting_storyline'
          && !selectionTouched.current
        ) {
          setSelectedStorylineIds(getDefaultStorylineSelection(data.auto_story))
        }
        if (data.auto_story?.stories?.length && !planTouched.current) {
          setStories(data.auto_story.stories)
          setValidated(Boolean(
            data.auto_story.validation
            && data.auto_story.stories.every(story => story.validation),
          ))
        }
        if (data.auto_story?.recipe && !legacyTouched.current) {
          setLegacyRecipe(data.auto_story.recipe)
          setLegacyTimingOptions(data.auto_story.timing_options || null)
          setLegacyValidated(Boolean(data.auto_story.validation))
        }
        if (
          data.status === 'awaiting_recipe'
          && data.auto_story?.phase !== 'reconstructing'
        ) {
          setReconstructingStoryId('')
        }
        if (!TERMINAL_STATUSES.includes(data.status)) {
          const atUserCheckpoint = ['awaiting_storyline', 'awaiting_recipe'].includes(data.status)
          timeout = setTimeout(poll, atUserCheckpoint ? 5000 : 2000)
        }
      } catch (err) {
        if (active) {
          consecutiveFailures += 1
          setConnectionError(`Connection interrupted. Retrying in a few seconds: ${err.message}`)
          const retryDelay = Math.min(30000, 2000 * (2 ** Math.min(consecutiveFailures - 1, 4)))
          timeout = setTimeout(poll, retryDelay)
        }
      }
    }

    poll()
    return () => {
      active = false
      clearTimeout(timeout)
    }
  }, [jobId])

  const handleFile = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    if (file.size > 2 * 1024 ** 3) {
      setError('Episode videos must be 2 GB or smaller.')
      event.target.value = ''
      return
    }
    setSelectedFile(file)
    setUploadFilename('')
    setError('')
    setUploading(true)
    setUploadProgress(0)
    uploadController.current = new AbortController()
    try {
      const result = await uploadVideo(
        file,
        percent => setUploadProgress(percent),
        uploadController.current.signal,
      )
      setUploadFilename(result.filename)
    } catch (err) {
      setSelectedFile(null)
      if (err.name !== 'AbortError') setError(err.message)
    } finally {
      uploadController.current = null
      setUploading(false)
      event.target.value = ''
    }
  }

  const handleProjectImport = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    if (file.size > 12 * 1024 ** 3) {
      setError('Project archives must be 12 GB or smaller.')
      event.target.value = ''
      return
    }
    setImportingProject(true)
    setImportProgress(0)
    setImportUploadComplete(false)
    setError('')
    importController.current = new AbortController()
    try {
      const restored = await importAutoStoryProject(
        file,
        percent => setImportProgress(percent),
        importController.current.signal,
        () => setImportUploadComplete(true),
      )
      navigate(`/auto-story/${restored.id}`)
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message)
    } finally {
      importController.current = null
      setImportingProject(false)
      setImportUploadComplete(false)
      event.target.value = ''
    }
  }

  const cancelUpload = () => uploadController.current?.abort()
  const cancelProjectImport = () => importController.current?.abort()

  const startAutoStory = async (event) => {
    event.preventDefault()
    setError('')
    if (!uploadFilename || !selectedFile) {
      setError('Choose and upload an episode video first.')
      return
    }
    setCreating(true)
    try {
      const created = await createAutoStory({
        upload_filename: uploadFilename,
        source_name: selectedFile.name,
        workflow_mode: workflowMode,
        target_duration_sec: Number(targetDuration),
        min_gap_sec: Number(minGapSec),
        diarization_enabled: diarizationEnabled,
        diarization_speakers: diarizationSpeakers === 'auto' ? null : Number(diarizationSpeakers),
        whisper_model: whisperModel,
        ai_provider: provider,
      })
      navigate(`/auto-story/${created.id}`)
    } catch (err) {
      setError(err.message)
    } finally {
      setCreating(false)
    }
  }

  const chooseStoryline = async (storylineId) => {
    setSelecting(true)
    setError('')
    try {
      const updated = await selectAutoStoryline(jobId, storylineId)
      setJob(updated)
      legacyTouched.current = false
      setLegacyRecipe(null)
      setLegacyTimingOptions(null)
      setLegacyValidated(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setSelecting(false)
    }
  }

  const toggleStoryline = (storylineId) => {
    selectionTouched.current = true
    setSelectedStorylineIds(current => (
      current.includes(storylineId)
        ? current.filter(id => id !== storylineId)
        : [...current, storylineId]
    ))
  }

  const createSelectedStories = async () => {
    if (!selectedStorylineIds.length) return
    setSelecting(true)
    setError('')
    try {
      const updated = await selectAutoStorylines(jobId, selectedStorylineIds, reviewBeforeRender)
      setJob(updated)
    } catch (err) {
      setError(err.message)
    } finally {
      setSelecting(false)
    }
  }

  const rerunStorylineDiscovery = async () => {
    setRediscovering(true)
    setError('')
    try {
      selectionTouched.current = false
      setSelectedStorylineIds([])
      const selectedProvider = retryProvider || job?.config?.ai_provider || 'gemini'
      const updated = await rediscoverAutoStorylines(jobId, selectedProvider)
      setJob(updated)
      setRetryProvider(selectedProvider)
    } catch (err) {
      setError(err.message)
    } finally {
      setRediscovering(false)
    }
  }

  const saveSpeakerNames = async () => {
    setRediscovering(true)
    setError('')
    try {
      selectionTouched.current = false
      setJob(await saveAutoStorySpeakerNames(
        jobId,
        speakerNameDraft,
        speakerCorrections,
        speakerMergeDraft,
      ))
      speakerNamesTouched.current = false
      setSpeakerCorrections([])
      setSpeakerTranscriptTurns([])
      setSpeakerTranscriptOpen(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setRediscovering(false)
    }
  }

  const loadSpeakerTranscript = async (offset = 0) => {
    setSpeakerTranscriptLoading(true)
    setError('')
    try {
      const result = await fetchAutoStorySpeakerTranscript(jobId, offset)
      setSpeakerTranscriptTurns(current => offset ? [...current, ...result.turns] : result.turns)
      setSpeakerTranscriptOffset(offset + result.turns.length)
      setSpeakerTranscriptTotal(result.total)
      setSpeakerTranscriptOpen(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setSpeakerTranscriptLoading(false)
    }
  }

  const updateSpeakerTurn = (turn, value) => {
    const correction = {
      start_word_id: turn.start_word_id,
      end_word_id: turn.end_word_id,
      speaker_id: value || null,
    }
    setSpeakerCorrections(current => [
      ...current.filter(item => (
        item.start_word_id !== correction.start_word_id
        || item.end_word_id !== correction.end_word_id
      )),
      correction,
    ])
    setSpeakerTranscriptTurns(current => current.map(item => (
      item.start_word_id === turn.start_word_id && item.end_word_id === turn.end_word_id
        ? { ...item, speaker_id: correction.speaker_id }
        : item
    )))
  }

  const previewSourceInterval = (startSec, endSec, key) => {
    const audio = speakerPreviewRef.current
    if (!audio) return
    const start = Math.max(0, Number(startSec) || 0)
    const end = Math.max(start, Number(endSec) || start) + 0.2
    setActiveSpeakerPreview(key)
    audio.pause()
    const startPlayback = () => {
      audio.currentTime = start
      audio.play().catch(err => {
        setActiveSpeakerPreview('')
        setError(`Could not play this source interval: ${err.message}`)
      })
    }
    if (audio.readyState >= 1) startPlayback()
    else audio.addEventListener('loadedmetadata', startPlayback, { once: true })
    audio.dataset.previewEnd = String(end)
  }

  const previewSpeakerTurn = turn => previewSourceInterval(
    turn.start_sec,
    turn.end_sec,
    `${turn.start_word_id}-${turn.end_word_id}`,
  )

  const stopSpeakerPreviewAtTurnEnd = () => {
    const audio = speakerPreviewRef.current
    const end = Number(audio?.dataset.previewEnd)
    if (audio && Number.isFinite(end) && audio.currentTime >= end) {
      audio.pause()
      setActiveSpeakerPreview('')
    }
  }

  const retryDiscoveryFromCache = async () => {
    setRetryingDiscovery(true)
    setError('')
    try {
      setJob(await retryAutoStoryDiscovery(jobId))
    } catch (err) {
      setError(err.message)
    } finally {
      setRetryingDiscovery(false)
    }
  }

  const retryRenderFromPlan = async () => {
    setRetryingRender(true)
    setError('')
    try {
      setJob(await retryAutoStoryRender(jobId))
    } catch (err) {
      setError(err.message)
    } finally {
      setRetryingRender(false)
    }
  }

  const retryFailedStory = async (storylineId) => {
    setRetryingStorylineId(storylineId)
    setError('')
    try {
      setJob(await retryAutoStoryline(jobId, storylineId))
    } catch (err) {
      setError(err.message)
      setRetryingStorylineId('')
    }
  }

  const editCompletedProject = async () => {
    setReopeningReview(true)
    setError('')
    try {
      setJob(await reopenAutoStoryReview(jobId))
      planTouched.current = false
      setValidated(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setReopeningReview(false)
    }
  }

  const editLegacyBeat = (beatIndex, key, value) => {
    legacyTouched.current = true
    setLegacyValidated(false)
    setError('')
    setLegacyRecipe(current => ({
      ...current,
      beats: current.beats.map((beat, index) => (
        index === beatIndex ? { ...beat, [key]: value } : beat
      )),
    }))
  }

  const getLegacyAdjacentBoundary = (beatIndex, key, direction) => {
    const beat = legacyRecipe?.beats?.[beatIndex]
    const options = legacyTimingOptions?.[String(beat?.beat_id ?? beatIndex + 1)]?.[key] || []
    const field = key === 'start' ? 'dialogue_start_sec' : 'dialogue_end_sec'
    const currentTime = Number(beat?.[field])
    if (!Number.isFinite(currentTime)) return null
    const candidates = options.filter(option => (
      direction < 0 ? option.time_sec < currentTime - 0.005 : option.time_sec > currentTime + 0.005
    ))
    return direction < 0 ? candidates[candidates.length - 1] || null : candidates[0] || null
  }

  const getLegacyBoundaryWord = (beat, beatIndex, key) => {
    const options = legacyTimingOptions?.[String(beat.beat_id ?? beatIndex + 1)]?.[key] || []
    const field = key === 'start' ? 'dialogue_start_sec' : 'dialogue_end_sec'
    const currentTime = Number(beat[field])
    if (!Number.isFinite(currentTime)) return null
    const nearest = options.reduce((best, option) => (
      !best || Math.abs(option.time_sec - currentTime) < Math.abs(best.time_sec - currentTime)
        ? option
        : best
    ), null)
    return nearest && Math.abs(nearest.time_sec - currentTime) <= 0.05 ? nearest.word : null
  }

  const moveLegacyBoundary = (beatIndex, key, direction) => {
    const beat = legacyRecipe?.beats?.[beatIndex]
    const next = getLegacyAdjacentBoundary(beatIndex, key, direction)
    if (!beat || !next) return
    const options = legacyTimingOptions?.[String(beat.beat_id ?? beatIndex + 1)]?.[key] || []
    const idField = key === 'start' ? 'start_word_id' : 'end_word_id'
    const currentIndex = options.findIndex(option => option.word_id === beat[idField])
    const nextIndex = options.findIndex(option => option.word_id === next.word_id)
    let textWords = String(beat.text || '').split(/\s+/).filter(Boolean)

    if (currentIndex >= 0 && nextIndex >= 0) {
      if (key === 'start' && nextIndex < currentIndex) {
        textWords = [...options.slice(nextIndex, currentIndex).map(option => option.word), ...textWords]
      } else if (key === 'start' && nextIndex > currentIndex) {
        textWords = textWords.slice(nextIndex - currentIndex)
      } else if (key === 'end' && nextIndex > currentIndex) {
        textWords = [...textWords, ...options.slice(currentIndex + 1, nextIndex + 1).map(option => option.word)]
      } else if (key === 'end' && nextIndex < currentIndex) {
        textWords = textWords.slice(0, Math.max(0, textWords.length - (currentIndex - nextIndex)))
      }
    }
    const field = key === 'start' ? 'dialogue_start_sec' : 'dialogue_end_sec'
    legacyTouched.current = true
    setLegacyValidated(false)
    setError('')
    setLegacyRecipe(current => ({
      ...current,
      beats: current.beats.map((item, index) => (
        index === beatIndex
          ? {
            ...item,
            [field]: next.time_sec,
            [idField]: next.word_id,
            text: textWords.join(' '),
          }
          : item
      )),
    }))
  }

  const validateLegacyRecipe = async () => {
    setValidating(true)
    setError('')
    try {
      const updated = await validateAutoStoryRecipe(jobId, legacyRecipe)
      setJob(updated)
      setLegacyRecipe(updated.auto_story.recipe)
      legacyTouched.current = false
      setLegacyValidated(true)
    } catch (err) {
      setError(err.message)
      setLegacyValidated(false)
    } finally {
      setValidating(false)
    }
  }

  const editBeat = (storyIndex, beatIndex, key, value) => {
    planTouched.current = true
    setValidated(false)
    setError('')
    setStories(current => current.map((story, index) => (
      index === storyIndex
        ? {
          ...story,
          recipe: {
            ...story.recipe,
            beats: story.recipe.beats.map((beat, currentBeatIndex) => (
              currentBeatIndex === beatIndex ? { ...beat, [key]: value } : beat
            )),
          },
        }
        : story
    )))
  }

  const getAdjacentBoundary = (storyIndex, beatIndex, key, direction) => {
    const story = stories[storyIndex]
    const beat = story?.recipe?.beats?.[beatIndex]
    const options = story?.timing_options?.[String(beat?.beat_id ?? beatIndex + 1)]?.[key] || []
    const currentTime = Number(beat?.[key === 'start' ? 'dialogue_start_sec' : 'dialogue_end_sec'])
    if (!Number.isFinite(currentTime)) return null

    const candidates = options.filter(option => (
      direction < 0 ? option.time_sec < currentTime - 0.005 : option.time_sec > currentTime + 0.005
    ))
    return direction < 0 ? candidates[candidates.length - 1] || null : candidates[0] || null
  }

  const getBoundaryWord = (storyIndex, beat, beatIndex, key) => {
    const options = stories[storyIndex]?.timing_options?.[
      String(beat.beat_id ?? beatIndex + 1)
    ]?.[key] || []
    const field = key === 'start' ? 'dialogue_start_sec' : 'dialogue_end_sec'
    const currentTime = Number(beat[field])
    if (!Number.isFinite(currentTime)) return null
    const nearest = options.reduce((best, option) => (
      !best || Math.abs(option.time_sec - currentTime) < Math.abs(best.time_sec - currentTime)
        ? option
        : best
    ), null)
    return nearest && Math.abs(nearest.time_sec - currentTime) <= 0.05 ? nearest.word : null
  }

  const moveBeatBoundary = (storyIndex, beatIndex, key, direction) => {
    const story = stories[storyIndex]
    const next = getAdjacentBoundary(storyIndex, beatIndex, key, direction)
    if (!next) return

    const beat = story.recipe.beats[beatIndex]
    const options = story.timing_options?.[String(beat.beat_id ?? beatIndex + 1)]?.[key] || []
    const idField = key === 'start' ? 'start_word_id' : 'end_word_id'
    const currentIndex = options.findIndex(option => option.word_id === beat[idField])
    const nextIndex = options.findIndex(option => option.word_id === next.word_id)
    let textWords = String(beat.text || '').split(/\s+/).filter(Boolean)

    if (currentIndex >= 0 && nextIndex >= 0) {
      if (key === 'start' && nextIndex < currentIndex) {
        textWords = [
          ...options.slice(nextIndex, currentIndex).map(option => option.word),
          ...textWords,
        ]
      } else if (key === 'start' && nextIndex > currentIndex) {
        textWords = textWords.slice(nextIndex - currentIndex)
      } else if (key === 'end' && nextIndex > currentIndex) {
        textWords = [
          ...textWords,
          ...options.slice(currentIndex + 1, nextIndex + 1).map(option => option.word),
        ]
      } else if (key === 'end' && nextIndex < currentIndex) {
        textWords = textWords.slice(0, Math.max(0, textWords.length - (currentIndex - nextIndex)))
      }
    }

    const field = key === 'start' ? 'dialogue_start_sec' : 'dialogue_end_sec'
    planTouched.current = true
    setValidated(false)
    setError('')
    setStories(current => current.map((item, index) => (
      index === storyIndex
        ? {
          ...item,
          recipe: {
            ...item.recipe,
            beats: item.recipe.beats.map((currentBeat, currentBeatIndex) => (
              currentBeatIndex === beatIndex
                ? {
                  ...currentBeat,
                  [field]: next.time_sec,
                  [idField]: next.word_id,
                  text: textWords.join(' '),
                }
                : currentBeat
            )),
          },
        }
        : item
    )))
  }

  const validatePlan = async () => {
    setValidating(true)
    setError('')
    try {
      const submittedStories = stories.map(({ story_id, recipe }) => ({
        story_id,
        recipe,
      }))
      const updated = await validateAutoStoryPlan(jobId, submittedStories)
      setJob(updated)
      setStories(updated.auto_story.stories)
      planTouched.current = false
      setValidated(true)
    } catch (err) {
      setError(err.message)
      setValidated(false)
    } finally {
      setValidating(false)
    }
  }

  const reconstructStory = async (storyId) => {
    setReconstructingStoryId(storyId)
    setError('')
    try {
      const updated = await reconstructAutoStoryStory(jobId, storyId)
      setJob(updated)
      planTouched.current = false
      setValidated(Boolean(updated.auto_story?.validation))
    } catch (err) {
      setError(err.message)
      setReconstructingStoryId('')
    }
  }

  const approveRender = async () => {
    setRendering(true)
    setError('')
    try {
      setJob(await renderAutoStory(jobId))
    } catch (err) {
      setError(err.message)
    } finally {
      setRendering(false)
    }
  }

  const requestCancel = async () => {
    setCancelling(true)
    setError('')
    try {
      await cancelAutoStory(jobId)
      const updated = await fetchAutoStory(jobId)
      setJob(updated)
    } catch (err) {
      setError(err.message)
    } finally {
      setCancelling(false)
    }
  }

  if (!jobId) {
    const providerKeySet = provider === 'gemini'
      ? settings?.google_api_key_set
      : provider === 'nvidia'
        ? settings?.nvidia_api_key_set
        : settings?.freellmapi_api_key_set
    const readyForStart = Boolean(providerKeySet && (!diarizationEnabled || settings?.hf_token_set))
    return (
      <div className="fade-in">
        <div className="page-header">
          <div>
            <h2>Auto Story</h2>
            <p>Build one dialogue story automatically, or choose storylines for multiple stories.</p>
          </div>
        </div>

        <section className="card auto-story-card">
          <h3 className="card-title">Restore a saved project</h3>
          <p className="auto-story-help">
            Import an Auto Story project archive to restore its episode, transcript cache, plans and rendered files.
          </p>
          <input
            ref={projectImportRef}
            type="file"
            accept=".zip,application/zip"
            onChange={handleProjectImport}
            hidden
          />
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => projectImportRef.current?.click()}
            disabled={importingProject}
          >
            {importingProject ? <><span className="spinner"></span> Restoring project…</> : 'Import Auto Story project (.zip)'}
          </button>
          <p className="form-hint">Large archives may take time to upload; the progress bar tracks transfer to the backend.</p>
          {importingProject && (
            <TransferProgress
              label={importUploadComplete ? 'Archive uploaded; restoring project on backend…' : 'Uploading project archive'}
              percent={importProgress}
              onCancel={cancelProjectImport}
              cancellable={!importUploadComplete}
            />
          )}
        </section>

        <section className="card auto-story-card auto-story-project-library">
          <div className="auto-story-library-heading">
            <div>
              <h3 className="card-title">Saved Auto Story projects</h3>
              <p className="auto-story-help">Projects are stored on the connected backend. Use a backup to move them or keep them beyond a temporary runtime.</p>
            </div>
            <span className="auto-story-library-count">{savedProjectsTotal}</span>
          </div>
          <label className="auto-story-project-search">
            <span className="sr-only">Search saved projects</span>
            <input
              className="form-input"
              type="search"
              value={projectSearch}
              onChange={event => setProjectSearch(event.target.value)}
              placeholder="Search by project, file, ID, or status"
            />
          </label>
          {savedProjectsError && <p className="auto-story-warning" role="status">Could not load saved projects: {savedProjectsError}</p>}
          {projectStorageStatus && !projectStorageStatus.healthy && (
            <p className="auto-story-warning" role="alert">
              The backend cannot confirm that project records are saving. Download backups for projects you need to keep. Storage error: {projectStorageStatus.error_type || 'unknown'}.
              {projectStorageStatus.recovery_required
                ? ' The saved project index could not be read and is protected from being overwritten; recover a valid jobs.json and restart the backend before relying on local persistence.'
                : ' The backend will retry saving on its next update; keep portable backups until this warning clears.'}
            </p>
          )}
          {loadingProjects && savedProjects.length === 0 && <p className="auto-story-help">Loading saved projects…</p>}
          {!loadingProjects && !savedProjectsError && savedProjects.length === 0 && (
            <p className="auto-story-help">
              {projectSearch.trim() ? 'No projects match this search.' : 'No saved Auto Story projects yet.'}
            </p>
          )}
          <div className="auto-story-project-list">
            {savedProjects.map(project => {
              const canExport = !['queued', 'downloading', 'transcribing', 'analyzing', 'rendering'].includes(project.status)
              return (
                <article className="auto-story-project-row" key={project.id}>
                  <div className="auto-story-project-details">
                    <strong>{project.source_name || project.upload_filename || 'Auto Story project'}</strong>
                    <span>{project.id} · {project.story_count || 0} stories</span>
                    <span>{project.updated_at ? new Date(project.updated_at).toLocaleString() : 'Saved'}</span>
                  </div>
                  <div className="auto-story-project-actions">
                    <span className={`badge badge-${project.status}`}>{project.status.replace(/_/g, ' ')}</span>
                    <Link to={`/auto-story/${encodeURIComponent(project.id)}`} className="btn btn-secondary btn-sm">Open</Link>
                    {canExport && (
                      <a className="btn btn-ghost btn-sm" href={`/api/auto-story/${encodeURIComponent(project.id)}/project`}>
                        Backup
                      </a>
                    )}
                  </div>
                </article>
              )
            })}
          </div>
          {savedProjects.length < savedProjectsTotal && (
            <button
              type="button"
              className="btn btn-ghost"
              onClick={loadMoreProjects}
              disabled={loadingMoreProjects || loadingProjects}
            >
              {loadingMoreProjects ? 'Loading more projects…' : `Load more (${savedProjects.length} of ${savedProjectsTotal})`}
            </button>
          )}
        </section>

        {error && <div className="auto-story-error" role="alert">{error}</div>}

        <form onSubmit={startAutoStory}>
          <div className="card auto-story-card">
            <h3 className="card-title">1. Choose your episode</h3>
            <p className="auto-story-help">
            Select a video from your computer. It is uploaded to the connected Studio backend for processing.
            </p>
            <input
              ref={fileRef}
              type="file"
              accept="video/mp4,video/x-matroska,video/quicktime,video/x-msvideo,video/webm,video/mp2t"
              onChange={handleFile}
              hidden
            />
            <div className="auto-story-file-row">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => fileRef.current?.click()}
                disabled={uploading}
              >
                {uploading ? <><span className="spinner"></span> Uploading episode…</> : '📁 Choose episode video'}
              </button>
              {selectedFile && !uploading && (
                <span className="auto-story-file-name">
                  {uploadFilename ? '✅ ' : '⏳ '}{selectedFile.name}
                </span>
              )}
            </div>
            {uploading && (
              <TransferProgress
                label="Uploading episode"
                percent={uploadProgress}
                onCancel={cancelUpload}
              />
            )}
            <p className="form-hint">MP4, MKV, AVI, MOV, WebM (maximum 2 GB).</p>
          </div>

          <div className="card auto-story-card">
            <div className="form-group">
              <label className="form-label" htmlFor="auto-story-workflow-mode">Story workflow</label>
              <select
                id="auto-story-workflow-mode"
                className="form-select"
                value={workflowMode}
                onChange={event => setWorkflowMode(event.target.value)}
              >
                <option value="auto">Auto Story · create one best story (recommended)</option>
                <option value="choose">Choose a Story · select storylines and create multiple</option>
              </select>
              <span className="form-hint">
                Auto Story creates one story from the strongest supported narrative. Choose a Story keeps the multi-story workflow.
              </span>
            </div>
            <h3 className="card-title">2. Set your story length</h3>
            <div className="auto-story-options">
              <div className="form-group">
                <label className="form-label" htmlFor="auto-story-duration">Target duration per story</label>
                <select
                  id="auto-story-duration"
                  className="form-select"
                  value={targetDuration}
                  onChange={event => setTargetDuration(event.target.value)}
                >
                  <option value={60}>About 60 seconds</option>
                  <option value={90}>About 90 seconds</option>
                  <option value={120}>About 2 minutes</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="auto-story-provider">AI provider</label>
                <select
                  id="auto-story-provider"
                  className="form-select"
                  value={provider}
                  onChange={event => setProvider(event.target.value)}
                >
                  <option value="gemini">Google Gemini</option>
                  <option value="nvidia">NVIDIA NIM</option>
                  <option value="freellmapi">FreeLLMAPI (unified key)</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="auto-story-min-gap">Minimum gap between moments</label>
                <select
                  id="auto-story-min-gap"
                  className="form-select"
                  value={minGapSec}
                  onChange={event => setMinGapSec(event.target.value)}
                >
                  {[15, 20, 25, 30, 40, 60].map(seconds => (
                    <option value={seconds} key={seconds}>{seconds} seconds</option>
                  ))}
                </select>
                <span className="form-hint">
                  Controls how far apart selected moments should be in the episode. Shorter gaps can create denser stories; 30 seconds is a balanced default.
                </span>
              </div>
              <div className="form-group">
                <label className="form-checkbox-label">
                  <input
                    type="checkbox"
                    checked={diarizationEnabled}
                    onChange={event => setDiarizationEnabled(event.target.checked)}
                  />
                  <span>Identify speakers with PyAnnote (optional; takes longer)</span>
                </label>
                <span className="form-hint">
                  Story selection works from the dialogue transcript without speaker identities. Enable this to label speakers and review speaker specific scenes.
                </span>
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="auto-story-speaker-count">Expected speaker count</label>
                <select
                  id="auto-story-speaker-count"
                  className="form-select"
                  value={diarizationSpeakers}
                  onChange={event => setDiarizationSpeakers(event.target.value)}
                  disabled={!diarizationEnabled}
                >
                  <option value="auto">Automatic (recommended)</option>
                  {Array.from({ length: 16 }, (_, index) => index + 1).map(count => (
                    <option value={count} key={count}>{count} {count === 1 ? 'speaker' : 'speakers'}</option>
                  ))}
                </select>
                <span className="form-hint">
                  Keep Automatic unless you know the exact count. A fixed count helps when speakers overlap or sound similar.
                </span>
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="auto-story-whisper-model">Transcription model</label>
                <select
                  id="auto-story-whisper-model"
                  className="form-select"
                  value={whisperModel}
                  onChange={event => setWhisperModel(event.target.value)}
                >
                  <option value="tiny">Tiny · fastest, lower accuracy</option>
                  <option value="base">Base · fast</option>
                  <option value="small">Small · balanced (default)</option>
                  <option value="medium">Medium · slower, more accurate</option>
                  <option value="large-v3">Large v3 · highest accuracy, most resources</option>
                  <option value="large-v3-turbo">Large v3 Turbo · high accuracy, GPU recommended</option>
                </select>
                <span className="form-hint">
                  Larger models use more memory and take longer. Device and compute type remain automatic for this backend.
                </span>
              </div>
            </div>
            <div className="auto-story-note">
              {workflowMode === 'auto'
                ? 'The AI finds one promising narrative, selects connected dialogue, and renders one story automatically.'
                : 'The AI discovers several storylines. Choose which to create; the backend builds and renders each selected story.'}
              {' '}Speaker identification is optional. The 9:16 renderer tracks the visible subject when possible.
            </div>
            {settingsError && <p className="auto-story-error">{settingsError}</p>}
            {settings && !readyForStart && (
              <div className="auto-story-warning">
                Before starting, add the selected AI provider key{diarizationEnabled && ' and Hugging Face token'} in{' '}
                <Link to="/settings">Studio Settings</Link>{diarizationEnabled && <> and accept the{' '}
                  <a href="https://huggingface.co/pyannote/speaker-diarization-3.1" target="_blank" rel="noopener noreferrer">
                    PyAnnote model terms
                  </a>.</>}
              </div>
            )}
          </div>

          {error && <div className="auto-story-error" role="alert">{error}</div>}
          <button
            type="submit"
            className="btn btn-primary"
            disabled={creating || uploading || !uploadFilename || !readyForStart}
          >
            {creating ? <><span className="spinner"></span> Starting…</> : workflowMode === 'auto' ? '✨ Create my story' : '✨ Find storylines'}
          </button>
        </form>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="empty-state">
        {error ? <><h3>Could not load Auto Story</h3><p>{error}</p></> : <><div className="spinner"></div><p>Loading Auto Story…</p></>}
        <Link to="/auto-story" className="btn btn-secondary">Start a new Auto Story</Link>
      </div>
    )
  }

  const state = job.auto_story || {}
  const canonicalSpeakerIds = (state.speaker_ids || []).filter(
    speakerId => !speakerMergeDraft[speakerId],
  )
  const speakerEditsDirty = Boolean(
    speakerNamesTouched.current
    || speakerCorrections.length
    || JSON.stringify(speakerMergeDraft) !== JSON.stringify(state.speaker_merges || {}),
  )
  const active = ['queued', 'downloading', 'transcribing', 'analyzing', 'rendering']
    .includes(job.status)
  const progress = job.progress?.percent || 0
  const canRetryRender = Boolean(
    (state.stories?.length && state.validation && state.stories.every(story => story.validation))
    || (state.recipe && state.validation && state.selected_storyline_id),
  )

  return (
    <div className="fade-in">
      <div className="page-header">
        <div>
          <h2>Auto Story-Recipe</h2>
          <p>
            {state.source_name || job.upload_filename} · about {state.target_duration_sec || 60} seconds per story
            {state.story_count ? ` · ${state.story_count} storylines found` : ''}
          </p>
        </div>
        <div className="auto-story-header-actions">
          <span className={`badge badge-${job.status}`}>{job.status.replace(/_/g, ' ')}</span>
          <a
            className="btn btn-ghost btn-sm"
            href={`/api/auto-story/${encodeURIComponent(jobId)}/diagnostics`}
            download
          >
            Download diagnostics
          </a>
          <Link to="/" className="btn btn-ghost btn-sm">← Dashboard</Link>
        </div>
      </div>

      {active && (
        <div className="card auto-story-progress">
          <div className="auto-story-progress-heading">
            <strong>{job.progress?.message || state.message || 'Preparing Auto Story…'}</strong>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="progress-bar-bg">
            <div className="progress-bar-fill" style={{ width: `${progress}%` }}></div>
          </div>
          <p className="auto-story-help">
            Long episodes can take a while to transcribe. You can return to this page while the backend continues working. Download a recovery backup before a temporary runtime shuts down.
          </p>
          {job.status !== 'rendering' && (
            <a className="btn btn-secondary btn-sm" href={`/api/auto-story/${encodeURIComponent(jobId)}/project`}>
              Download recovery backup
            </a>
          )}
          <button type="button" className="btn btn-danger btn-sm" onClick={requestCancel} disabled={cancelling}>
            {cancelling ? 'Requesting…' : 'Cancel'}
          </button>
        </div>
      )}

      {error && <div className="auto-story-error" role="alert">{error}</div>}
      {connectionError && <div className="auto-story-warning" role="status">{connectionError}</div>}
      {job.persistence && !job.persistence.healthy && (
        <div className="auto-story-warning" role="alert">
          The backend cannot confirm that this project is saving to disk. Download a portable backup as soon as the current stage reaches a safe checkpoint.
          {job.persistence.recovery_required
            ? ' Its existing job index could not be read and will not be replaced automatically; restore a valid jobs.json before relying on backend persistence.'
            : ` Save error: ${job.persistence.error_type || 'unknown'}.`}
        </div>
      )}
      {job.error && <div className="auto-story-error" role="alert">{job.error}</div>}

      {job.status === 'awaiting_storyline' && (
        <section className="auto-story-section">
          <h3>{state.last_failed_stage ? 'Story discovery needs a retry' : 'Choose which stories to create'}</h3>
          <a className="btn btn-secondary" href={`/api/auto-story/${encodeURIComponent(jobId)}/project`}>
            Download project backup
          </a>
          {state.last_failed_stage && (
            <>
              <div className="auto-story-warning" role="alert">
                {state.message || 'The AI provider request failed. Your transcript and speaker analysis remain saved.'}
              </div>
              {state.last_failed_stage === 'storyline_discovery' && (
                <div className="form-group">
                  <label className="form-label" htmlFor="auto-story-retry-provider">AI provider for retry</label>
                  <select
                    id="auto-story-retry-provider"
                    className="form-select"
                    value={retryProvider || job.config?.ai_provider || 'gemini'}
                    onChange={event => setRetryProvider(event.target.value)}
                  >
                    <option value="gemini" disabled={Boolean(settings && !settings.google_api_key_set)}>
                      Gemini{settings && !settings.google_api_key_set ? ' · API key not configured' : ''}
                    </option>
                    <option value="nvidia" disabled={Boolean(settings && !settings.nvidia_api_key_set)}>
                      NVIDIA NIM{settings && !settings.nvidia_api_key_set ? ' · API key not configured' : ''}
                    </option>
                    <option value="freellmapi" disabled={Boolean(settings && !settings.freellmapi_api_key_set)}>
                      FreeLLMAPI{settings && !settings.freellmapi_api_key_set ? ' · API key not configured' : ''}
                    </option>
                  </select>
                  <span className="form-hint">
                    Retrying uses the saved transcript and speaker analysis; it does not rerun transcription or diarization.
                    {' '}<Link to="/settings">Manage provider keys</Link>.
                  </span>
                </div>
              )}
            </>
          )}
          {state.storylines?.length > 0 && <p className="auto-story-help">
            Select one or more storylines. All are selected initially; uncheck any you do not want.
            The backend automatically creates and renders them unless you choose manual timing review below.
          </p>}
          {!!state.speaker_ids?.length && (
            <div className="auto-story-speaker-review">
              <h4>Who is speaking?</h4>
              <p className="auto-story-help">
                Diarization groups words by detected voice and assigns labels such as SPEAKER_00. Name speakers, merge duplicate labels for one person, and correct individual transcript turns before creating stories.
              </p>
              <div className="auto-story-speaker-fields">
                {state.speaker_ids.map((speakerId, index) => {
                  let mergedInto = speakerMergeDraft[speakerId] || ''
                  while (speakerMergeDraft[mergedInto]) mergedInto = speakerMergeDraft[mergedInto]
                  return (
                    <div className="auto-story-speaker-edit" key={speakerId}>
                      <span>{speakerId}</span>
                      {mergedInto
                        ? <span className="auto-story-speaker-merge-note">Merged into {mergedInto}</span>
                        : <input
                          className="form-input"
                          maxLength={60}
                          placeholder={`Speaker ${index + 1} (optional name)`}
                          value={speakerNameDraft[speakerId] || ''}
                          onChange={event => {
                            speakerNamesTouched.current = true
                            setSpeakerNameDraft(current => ({ ...current, [speakerId]: event.target.value }))
                          }}
                        />}
                      <select
                        className="form-select"
                        aria-label={`Merge ${speakerId} into another speaker`}
                        value={mergedInto}
                        onChange={event => {
                          speakerNamesTouched.current = true
                          setSpeakerMergeDraft(current => {
                            const next = { ...current }
                            if (event.target.value) next[speakerId] = event.target.value
                            else delete next[speakerId]
                            return next
                          })
                        }}
                      >
                        <option value="">Keep separate</option>
                        {canonicalSpeakerIds.filter(target => target !== speakerId).map(target => (
                          <option value={target} key={target}>Merge into {target}</option>
                        ))}
                      </select>
                    </div>
                  )
                })}
              </div>
              {speakerEditsDirty && (
                <p className="auto-story-warning" role="status">
                  Save these speaker edits before selecting storylines; otherwise they will not be applied.
                </p>
              )}
              <button
                type="button"
                className="btn btn-secondary"
                onClick={saveSpeakerNames}
                disabled={rediscovering || selecting || !speakerEditsDirty}
              >
                {rediscovering
                  ? 'Updating storylines…'
                  : speakerCorrections.length
                    ? 'Save speaker corrections & refresh storylines'
                    : 'Save speaker names & refresh storylines'}
              </button>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => (speakerTranscriptOpen
                  ? setSpeakerTranscriptOpen(false)
                  : loadSpeakerTranscript(0))}
                disabled={speakerTranscriptLoading || rediscovering}
              >
                {speakerTranscriptLoading
                  ? 'Loading transcript…'
                  : speakerTranscriptOpen
                    ? 'Hide speaker turn review'
                    : 'Review & correct transcript speaker turns'}
              </button>
              {speakerTranscriptOpen && (
                <div className="auto-story-transcript-review">
                  <p className="auto-story-help">
                    Each row groups neighboring transcript words. Change a row to reassign that complete dialogue turn; changes are saved in this project and used for story discovery and rendering.
                  </p>
                  <label className="auto-story-audio-preview">
                    <span>Episode audio for turn preview</span>
                    <audio
                      ref={speakerPreviewRef}
                      controls
                      preload="metadata"
                      src={`/api/auto-story/${encodeURIComponent(jobId)}/source`}
                      onTimeUpdate={stopSpeakerPreviewAtTurnEnd}
                      onEnded={() => setActiveSpeakerPreview('')}
                      onError={() => {
                        setActiveSpeakerPreview('')
                        setError('Could not load the uploaded episode audio. Restore the project source and try again.')
                      }}
                    />
                  </label>
                  <div className="auto-story-turn-list">
                    {speakerTranscriptTurns.map(turn => {
                      let displayedSpeakerId = turn.speaker_id || ''
                      while (speakerMergeDraft[displayedSpeakerId]) {
                        displayedSpeakerId = speakerMergeDraft[displayedSpeakerId]
                      }
                      return (
                        <div className="auto-story-turn-row" key={`${turn.start_word_id}-${turn.end_word_id}`}>
                          <span className="auto-story-turn-time">
                            {turn.start_sec.toFixed(1)}s–{turn.end_sec.toFixed(1)}s
                          </span>
                          <span className="auto-story-turn-text">{turn.text}</span>
                          <select
                            className="form-select"
                            aria-label={`Assign speaker for transcript turn starting ${turn.start_sec.toFixed(1)} seconds`}
                            value={displayedSpeakerId}
                            onChange={event => updateSpeakerTurn(turn, event.target.value)}
                          >
                            <option value="">Unassigned</option>
                            {canonicalSpeakerIds.map(speakerId => (
                              <option value={speakerId} key={speakerId}>
                                {speakerNameDraft[speakerId] || speakerId}
                              </option>
                            ))}
                          </select>
                          <button
                            type="button"
                            className="btn btn-ghost btn-sm"
                            onClick={() => previewSpeakerTurn(turn)}
                            aria-label={`Play audio from ${turn.start_sec.toFixed(1)} to ${turn.end_sec.toFixed(1)} seconds`}
                          >
                            {activeSpeakerPreview === `${turn.start_word_id}-${turn.end_word_id}` ? 'Playing turn…' : 'Play turn'}
                          </button>
                        </div>
                      )
                    })}
                  </div>
                  {speakerTranscriptOffset < speakerTranscriptTotal && (
                    <button
                      type="button"
                      className="btn btn-ghost"
                      onClick={() => loadSpeakerTranscript(speakerTranscriptOffset)}
                      disabled={speakerTranscriptLoading}
                    >
                      Load more turns ({speakerTranscriptTurns.length} of {speakerTranscriptTotal})
                    </button>
                  )}
                  {!!speakerCorrections.length && (
                    <p className="auto-story-help" role="status">
                      {speakerCorrections.length} speaker turn edit(s) will be saved with this project.
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
          <label className="auto-story-review-choice">
            <input
              type="checkbox"
              checked={reviewBeforeRender}
              onChange={event => setReviewBeforeRender(event.target.checked)}
            />
            <span>Let me review and adjust dialogue timing before rendering</span>
          </label>
          {state.failed_stories?.length > 0 && (
            <div className="auto-story-warning" role="status">
              <strong>Some storylines failed automatic scheduling.</strong>
              <p>They are unchecked below. Choose different arcs or rerun discovery; any successful outputs are kept.</p>
              <ul>
                {state.failed_stories.map(story => (
                  <li key={story.storyline_id}>
                    <strong>{story.title}:</strong> {story.reason}
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm"
                      onClick={() => retryFailedStory(story.storyline_id)}
                      disabled={Boolean(retryingStorylineId) || selecting || rediscovering}
                    >
                      {retryingStorylineId === story.storyline_id ? 'Retrying…' : 'Retry this story'}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div className="auto-story-storylines">
            {state.storylines?.map((storyline, index) => (
              <article className="card auto-story-storyline" key={storyline.id}>
                <label className="auto-story-storyline-choice">
                  <input
                    type="checkbox"
                    checked={selectedStorylineIds.includes(storyline.id)}
                    onChange={() => toggleStoryline(storyline.id)}
                    aria-label={`Select storyline ${index + 1}: ${storyline.title}`}
                  />
                  <span className="auto-story-kicker">Storyline {index + 1}</span>
                </label>
                <h4>{storyline.title}</h4>
                <p>{storyline.summary}</p>
                <div className="auto-story-meta">
                  <span>Characters: {storyline.key_characters?.join(', ') || 'Not specified'}</span>
                  <span>Speakers: {storyline.speaker_ids?.join(', ') || 'Not identified'}</span>
                  <span>
                    Episode span: {Math.floor(storyline.start_sec / 60)}:{String(Math.floor(storyline.start_sec % 60)).padStart(2, '0')}
                    {' – '}
                    {Math.floor(storyline.end_sec / 60)}:{String(Math.floor(storyline.end_sec % 60)).padStart(2, '0')}
                  </span>
                </div>
              </article>
            ))}
          </div>
          {state.storylines?.length === 0 && !state.last_failed_stage && (
            <div className="auto-story-warning">
              No storylines were found. Rerun discovery to ask the AI for a fresh set.
            </div>
          )}
          <div className="auto-story-actions">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={rerunStorylineDiscovery}
              disabled={rediscovering || selecting}
            >
              {rediscovering
                ? <><span className="spinner"></span> Retrying story discovery…</>
                : state.last_failed_stage
                  ? `Retry discovery with ${({ gemini: 'Gemini', nvidia: 'NVIDIA NIM', freellmapi: 'FreeLLMAPI' }[retryProvider || job.config?.ai_provider || 'gemini'])}`
                  : '🔄 Rerun story discovery'}
            </button>
            {state.storylines?.length > 0 && <button
              type="button"
              className="btn btn-primary"
              onClick={createSelectedStories}
              disabled={
                !selectedStorylineIds.length
                || selecting
                || rediscovering
                || speakerEditsDirty
              }
            >
              {selecting
                ? <><span className="spinner"></span> Creating stories…</>
                : reviewBeforeRender
                  ? `Create ${selectedStorylineIds.length} ${selectedStorylineIds.length === 1 ? 'story plan' : 'story plans'}`
                  : `✅ Create & render ${selectedStorylineIds.length} ${selectedStorylineIds.length === 1 ? 'story' : 'stories'}`}
            </button>}
          </div>
        </section>
      )}

      {job.status === 'awaiting_recipe' && !state.stories?.length && legacyRecipe && (
        <section className="auto-story-section">
          <h3>Review the dialogue beats</h3>
          <p className="auto-story-help">
            This job uses the earlier single-story workflow. Its schedule was prechecked; timing changes need a new validation.
          </p>
          <label className="auto-story-audio-preview">
            <span>Episode audio preview</span>
            <audio
              ref={speakerPreviewRef}
              controls
              preload="metadata"
              src={`/api/auto-story/${encodeURIComponent(jobId)}/source`}
              onTimeUpdate={stopSpeakerPreviewAtTurnEnd}
              onEnded={() => setActiveSpeakerPreview('')}
              onError={() => setError('Could not load the uploaded episode audio. Restore the project source and try again.')}
            />
          </label>
          <div className="auto-story-beats">
            {legacyRecipe.beats?.map((beat, beatIndex) => (
              <article className="card auto-story-beat" key={beat.beat_id || beatIndex}>
                <div>
                  <span className="auto-story-kicker">Beat {beatIndex + 1} · {beat.narrative_role || 'Dialogue'}</span>
                  <p className="auto-story-dialogue">“{beat.text}”</p>
                  <span className="auto-story-meta-text">
                    Speaker: {beat.speaker_ids?.join(', ') || 'Not identified'}
                  </span>
                </div>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={() => {
                    const padding = Number(legacyRecipe.settings?.padding_sec || 0)
                    previewSourceInterval(
                      Number(beat.dialogue_start_sec) - padding,
                      Number(beat.dialogue_end_sec) + padding,
                      `legacy-${beatIndex}`,
                    )
                  }}
                >
                  {activeSpeakerPreview === `legacy-${beatIndex}` ? 'Playing beat…' : 'Preview beat with context'}
                </button>
                <div className="auto-story-time-fields">
                  {[
                    { key: 'start', field: 'dialogue_start_sec', label: 'Start (seconds)' },
                    { key: 'end', field: 'dialogue_end_sec', label: 'End (seconds)' },
                  ].map(({ key, field, label }) => (
                    <label key={key}>
                      {label}
                      <input
                        className="form-input"
                        type="number"
                        step="0.01"
                        value={beat[field]}
                        onChange={event => editLegacyBeat(beatIndex, field, event.target.value)}
                      />
                      {legacyTimingOptions?.[String(beat.beat_id ?? beatIndex + 1)]?.[key] && (
                        <span className="auto-story-word-controls">
                          <button
                            type="button"
                            className="btn btn-ghost btn-sm"
                            onClick={() => moveLegacyBoundary(beatIndex, key, -1)}
                            disabled={!getLegacyAdjacentBoundary(beatIndex, key, -1)}
                            aria-label={`Move beat ${beatIndex + 1} ${key} boundary one word earlier`}
                          >
                            {key === 'start' ? '← Add word' : '← Remove word'}
                          </button>
                          <span className="auto-story-boundary-word">
                            {getLegacyBoundaryWord(beat, beatIndex, key) || 'Custom time'}
                          </span>
                          <button
                            type="button"
                            className="btn btn-ghost btn-sm"
                            onClick={() => moveLegacyBoundary(beatIndex, key, 1)}
                            disabled={!getLegacyAdjacentBoundary(beatIndex, key, 1)}
                            aria-label={`Move beat ${beatIndex + 1} ${key} boundary one word later`}
                          >
                            {key === 'start' ? 'Remove word →' : 'Add word →'}
                          </button>
                        </span>
                      )}
                    </label>
                  ))}
                </div>
              </article>
            ))}
          </div>
          <div className="auto-story-actions">
            {!legacyValidated && (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={validateLegacyRecipe}
                disabled={validating}
              >
                {validating ? 'Checking…' : 'Check my recipe'}
              </button>
            )}
            <button
              type="button"
              className="btn btn-primary"
              onClick={approveRender}
              disabled={!legacyValidated || rendering}
            >
              {rendering ? 'Starting render…' : '✅ Approve & render'}
            </button>
          </div>
        </section>
      )}

      {job.status === 'awaiting_recipe' && stories.length > 0 && (
        <section className="auto-story-section">
          <h3>3. Review the complete story plan</h3>
          <a className="btn btn-secondary" href={`/api/auto-story/${encodeURIComponent(jobId)}/project`}>
            Download project backup
          </a>
          <p className="auto-story-help">
            All {stories.length} narrative arcs and dialogue schedules have been created and checked before rendering.
            Review them together. Reconstruct any one story without changing the others, or adjust word-boundary timings.
          </p>
          <label className="auto-story-audio-preview">
            <span>Episode audio preview</span>
            <audio
              ref={speakerPreviewRef}
              controls
              preload="metadata"
              src={`/api/auto-story/${encodeURIComponent(jobId)}/source`}
              onTimeUpdate={stopSpeakerPreviewAtTurnEnd}
              onEnded={() => setActiveSpeakerPreview('')}
              onError={() => setError('Could not load the uploaded episode audio. Restore the project source and try again.')}
            />
          </label>
          <div className="auto-story-plan-list">
            {stories.map((story, storyIndex) => (
              <article className="card auto-story-storyline auto-story-plan" key={story.story_id}>
                <div className="auto-story-plan-heading">
                  <div>
                    <span className="auto-story-kicker">
                      Story {storyIndex + 1} of {stories.length} · {story.validation?.estimated_runtime_sec?.toFixed(1) || '—'}s planned
                      {story.settings?.best_effort && ' · Best effort; target not met'}
                    </span>
                    <h4>{story.storyline.title}</h4>
                    <p>{story.storyline.summary}</p>
                    <div className="auto-story-meta">
                      <span>Characters: {story.storyline.key_characters?.join(', ') || 'Not specified'}</span>
                      <span>Speakers: {story.storyline.speaker_ids?.join(', ') || 'Not identified'}</span>
                      <span>
                        Episode span: {Math.floor(story.storyline.start_sec / 60)}:{String(Math.floor(story.storyline.start_sec % 60)).padStart(2, '0')}
                        {' – '}
                        {Math.floor(story.storyline.end_sec / 60)}:{String(Math.floor(story.storyline.end_sec % 60)).padStart(2, '0')}
                      </span>
                    </div>
                  </div>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => reconstructStory(story.story_id)}
                    disabled={!validated || validating || Boolean(reconstructingStoryId) || rendering}
                  >
                    {reconstructingStoryId === story.story_id
                      ? <><span className="spinner"></span> Reconstructing…</>
                      : '🔁 Reconstruct story'}
                  </button>
                </div>
                <div className="auto-story-beats">
                  {story.recipe?.beats?.map((beat, beatIndex) => (
                    <article className="card auto-story-beat" key={beat.beat_id || beatIndex}>
                      <div className="auto-story-beat-heading">
                        <div>
                          <span className="auto-story-kicker">Beat {beatIndex + 1} · {beat.narrative_role || 'Dialogue'}</span>
                          <p className="auto-story-dialogue">“{beat.text}”</p>
                          <span className="auto-story-meta-text">
                            Speaker: {beat.speaker_ids?.join(', ') || 'Not identified'}
                          </span>
                        </div>
                      </div>
                      <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        onClick={() => {
                          const padding = Number(story.recipe.settings?.padding_sec || 0)
                          previewSourceInterval(
                            Number(beat.dialogue_start_sec) - padding,
                            Number(beat.dialogue_end_sec) + padding,
                            `${story.story_id}-${beatIndex}`,
                          )
                        }}
                      >
                        {activeSpeakerPreview === `${story.story_id}-${beatIndex}`
                          ? 'Playing beat…'
                          : 'Preview beat with context'}
                      </button>
                      <div className="auto-story-time-fields">
                        {[
                          { key: 'start', field: 'dialogue_start_sec', label: 'Start (seconds)' },
                          { key: 'end', field: 'dialogue_end_sec', label: 'End (seconds)' },
                        ].map(({ key, field, label }) => (
                          <label key={key}>
                            {label}
                            <input
                              className="form-input"
                              type="number"
                              step="0.01"
                              value={beat[field]}
                              onChange={event => editBeat(storyIndex, beatIndex, field, event.target.value)}
                            />
                            {story.timing_options?.[String(beat.beat_id ?? beatIndex + 1)]?.[key] && (
                              <span className="auto-story-word-controls">
                                <button
                                  type="button"
                                  className="btn btn-ghost btn-sm"
                                  onClick={() => moveBeatBoundary(storyIndex, beatIndex, key, -1)}
                                  disabled={!getAdjacentBoundary(storyIndex, beatIndex, key, -1)}
                                  aria-label={`Move story ${storyIndex + 1} beat ${beatIndex + 1} ${key} boundary one word earlier`}
                                >
                                  {key === 'start' ? '← Add word' : '← Remove word'}
                                </button>
                                <span
                                  className="auto-story-boundary-word"
                                  title={getBoundaryWord(storyIndex, beat, beatIndex, key) || 'Time is not on a cached word boundary'}
                                >
                                  {getBoundaryWord(storyIndex, beat, beatIndex, key) || 'Custom time'}
                                </span>
                                <button
                                  type="button"
                                  className="btn btn-ghost btn-sm"
                                  onClick={() => moveBeatBoundary(storyIndex, beatIndex, key, 1)}
                                  disabled={!getAdjacentBoundary(storyIndex, beatIndex, key, 1)}
                                  aria-label={`Move story ${storyIndex + 1} beat ${beatIndex + 1} ${key} boundary one word later`}
                                >
                                  {key === 'start' ? 'Remove word →' : 'Add word →'}
                                </button>
                              </span>
                            )}
                          </label>
                        ))}
                      </div>
                    </article>
                  ))}
                </div>
              </article>
            ))}
          </div>
          {state.validation && validated && (
            <div className="auto-story-success">
              All {state.validation.valid_story_count} stories are valid. Combined runtime:
              {' '}{state.validation.estimated_runtime_sec.toFixed(1)} seconds. Approve once to render them separately.
            </div>
          )}
          <div className="auto-story-actions">
            {!validated && (
              <button type="button" className="btn btn-secondary" onClick={validatePlan} disabled={validating}>
                {validating ? <><span className="spinner"></span> Checking…</> : 'Check the full plan'}
              </button>
            )}
            <button type="button" className="btn btn-primary" onClick={approveRender} disabled={!validated || rendering}>
              {rendering ? <><span className="spinner"></span> Starting render…</> : '✅ Approve & render all stories'}
            </button>
          </div>
        </section>
      )}

      {job.status === 'completed' && (
        <section className="auto-story-section">
          <h3>Your stories are ready</h3>
          <a className="btn btn-secondary" href={`/api/auto-story/${encodeURIComponent(jobId)}/project`}>
            Download portable project backup
          </a>
          {state.stories?.length > 0 && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={editCompletedProject}
              disabled={reopeningReview}
            >
              {reopeningReview ? 'Opening saved plan…' : 'Edit plans & render again'}
            </button>
          )}
          {state.failed_stories?.length > 0 && (
            <div className="auto-story-warning" role="status">
              <strong>
                Rendered {state.stories?.length || 0} of {state.selected_story_count || state.selected_storyline_ids?.length || 0} selected stories.
              </strong>
              <p>
                These storylines could not meet the timing and pacing checks, so they were skipped.
                Your successful renders are still available below.
              </p>
              <ul>
                {state.failed_stories.map(story => (
                  <li key={story.storyline_id}>
                    <strong>{story.title}:</strong> {story.reason}
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => retryFailedStory(story.storyline_id)}
                      disabled={Boolean(retryingStorylineId)}
                    >
                      {retryingStorylineId === story.storyline_id ? 'Retrying…' : 'Retry this story'}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {state.stories?.length ? state.stories.map((story, index) => (
            <article className="card auto-story-storyline auto-story-result" key={story.story_id}>
              <span className="auto-story-kicker">Story {index + 1} of {state.stories.length}</span>
              <h4>{story.storyline.title}</h4>
              <p>{story.storyline.summary}</p>
              {story.output_files?.some(file => file.filename.endsWith('.mp4')) && (
                <video
                  className="auto-story-video"
                  controls
                  preload="metadata"
                  src={story.output_files.find(file => file.filename.endsWith('.mp4')).download_url}
                />
              )}
              <div className="auto-story-downloads">
                {story.output_files?.map(file => (
                  <a className="btn btn-secondary" href={file.download_url} download key={file.filename}>
                    ⬇️ {file.filename}
                  </a>
                ))}
              </div>
            </article>
          )) : (
            <div className="auto-story-downloads">
              {state.output_files?.map(file => (
                <a className="btn btn-secondary" href={file.download_url} download key={file.filename}>
                  ⬇️ {file.filename}
                </a>
              ))}
            </div>
          )}
          {state.output_files?.filter(file => file.filename === 'story_plan_manifest.json').map(file => (
            <a className="btn btn-secondary" href={file.download_url} download key={file.filename}>
              ⬇️ Download complete plan manifest
            </a>
          ))}
        </section>
      )}

      {job.status === 'failed' && (
        <section className="auto-story-section">
          <a className="btn btn-secondary" href={`/api/auto-story/${encodeURIComponent(jobId)}/project`}>
            Download project backup
          </a>
          <button
            type="button"
            className="btn btn-primary"
            onClick={retryDiscoveryFromCache}
            disabled={retryingDiscovery}
          >
            {retryingDiscovery ? 'Retrying from saved analysis…' : 'Retry Auto Story (reuse saved analysis when available)'}
          </button>
          {canRetryRender && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={retryRenderFromPlan}
              disabled={retryingRender}
            >
              {retryingRender ? 'Resuming render…' : 'Retry rendering from saved plan'}
            </button>
          )}
          {state.failed_stories?.length > 0 && (
            <div className="auto-story-warning">
              <h3>Could not schedule the selected stories</h3>
              <p>The automatic pacing fallbacks were exhausted. Try rerunning discovery for a different set of storylines.</p>
              <ul>
                {state.failed_stories.map(story => (
                  <li key={story.storyline_id}>
                    <strong>{story.title}:</strong> {story.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {state.output_files?.length > 0 && (
            <div className="card auto-story-storyline auto-story-result">
              <h3>Some stories rendered before the failure</h3>
              <p>
                {state.rendered_story_count || 0} of {state.story_count || state.stories?.length || 0}
                {' '}stories completed. Download the available outputs before retrying.
              </p>
              <div className="auto-story-downloads">
                {state.output_files.map(file => (
                  <a className="btn btn-secondary" href={file.download_url} download key={file.filename}>
                    ⬇️ {file.filename}
                  </a>
                ))}
              </div>
            </div>
          )}
          <Link to="/auto-story" className="btn btn-primary">Start another Auto Story</Link>
        </section>
      )}

      {job.status === 'cancelled' && (
        <section className="auto-story-section">
          <h3>Processing was cancelled</h3>
          <p className="auto-story-help">
            Download a backup now, or continue with the saved transcript and speaker analysis when available.
          </p>
          <a className="btn btn-secondary" href={`/api/auto-story/${encodeURIComponent(jobId)}/project`}>
            Download project backup
          </a>
          <button
            type="button"
            className="btn btn-primary"
            onClick={retryDiscoveryFromCache}
            disabled={retryingDiscovery}
          >
            {retryingDiscovery ? 'Retrying…' : 'Continue project'}
          </button>
          {canRetryRender && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={retryRenderFromPlan}
              disabled={retryingRender}
            >
              {retryingRender ? 'Resuming render…' : 'Resume rendering from saved plan'}
            </button>
          )}
          {error && <div className="auto-story-error" role="alert">{error}</div>}
        </section>
      )}
    </div>
  )
}

export default AutoStory
