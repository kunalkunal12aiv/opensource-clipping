import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  cancelAutoStory,
  createAutoStory,
  fetchAutoStory,
  fetchSettings,
  rediscoverAutoStorylines,
  reconstructAutoStoryStory,
  renderAutoStory,
  selectAutoStoryline,
  selectAutoStorylines,
  uploadVideo,
  validateAutoStoryRecipe,
  validateAutoStoryPlan,
} from '../api'

const TERMINAL_STATUSES = ['completed', 'failed', 'cancelled']

function AutoStory() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const fileRef = useRef(null)
  const planTouched = useRef(false)
  const selectionTouched = useRef(false)
  const displayedDiscoveryRevision = useRef(null)
  const [settings, setSettings] = useState(null)
  const [settingsError, setSettingsError] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploadFilename, setUploadFilename] = useState('')
  const [uploading, setUploading] = useState(false)
  const [targetDuration, setTargetDuration] = useState(60)
  const [provider, setProvider] = useState('gemini')
  const [creating, setCreating] = useState(false)
  const [job, setJob] = useState(null)
  const [stories, setStories] = useState([])
  const [legacyRecipe, setLegacyRecipe] = useState(null)
  const [legacyTimingOptions, setLegacyTimingOptions] = useState(null)
  const [validated, setValidated] = useState(false)
  const [legacyValidated, setLegacyValidated] = useState(false)
  const [validating, setValidating] = useState(false)
  const [selectedStorylineIds, setSelectedStorylineIds] = useState([])
  const [selecting, setSelecting] = useState(false)
  const [rediscovering, setRediscovering] = useState(false)
  const [reconstructingStoryId, setReconstructingStoryId] = useState('')
  const [rendering, setRendering] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [error, setError] = useState('')
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
    if (!jobId) return undefined
    let active = true
    let timeout

    const poll = async () => {
      try {
        const data = await fetchAutoStory(jobId)
        if (!active) return
        setJob(data)
        const currentDiscoveryRevision = data.auto_story?.discovery_revision ?? 0
        if (
          data.status === 'awaiting_storyline'
          && displayedDiscoveryRevision.current !== currentDiscoveryRevision
        ) {
          displayedDiscoveryRevision.current = currentDiscoveryRevision
          selectionTouched.current = false
          setSelectedStorylineIds(
            (data.auto_story.storylines || []).map(storyline => storyline.id),
          )
        } else if (
          data.status === 'awaiting_storyline'
          && !selectionTouched.current
        ) {
          setSelectedStorylineIds(
            (data.auto_story.storylines || []).map(storyline => storyline.id),
          )
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
          timeout = setTimeout(poll, 2000)
        }
      } catch (err) {
        if (active) setError(err.message)
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
    setSelectedFile(file)
    setUploadFilename('')
    setError('')
    setUploading(true)
    try {
      const result = await uploadVideo(file)
      setUploadFilename(result.filename)
    } catch (err) {
      setSelectedFile(null)
      setError(err.message)
    } finally {
      setUploading(false)
      event.target.value = ''
    }
  }

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
        target_duration_sec: Number(targetDuration),
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
      const updated = await selectAutoStorylines(jobId, selectedStorylineIds)
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
      const updated = await rediscoverAutoStorylines(jobId)
      setJob(updated)
    } catch (err) {
      setError(err.message)
    } finally {
      setRediscovering(false)
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
      : settings?.nvidia_api_key_set
    const readyForStart = Boolean(settings?.hf_token_set && providerKeySet)

    return (
      <div className="fade-in">
        <div className="page-header">
          <div>
            <h2>Auto Story</h2>
            <p>Choose storylines once; Auto Story builds and renders them automatically.</p>
          </div>
        </div>

        <form onSubmit={startAutoStory}>
          <div className="card auto-story-card">
            <h3 className="card-title">1. Choose your episode</h3>
            <p className="auto-story-help">
              Select a video from your computer. It stays on this machine and is uploaded to your local Studio.
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
            <p className="form-hint">MP4, MKV, AVI, MOV, WebM (maximum 2 GB).</p>
          </div>

          <div className="card auto-story-card">
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
                </select>
              </div>
            </div>
            <div className="auto-story-note">
              AI discovers the storylines first. You choose which ones to create; then the backend automatically
              builds, checks, and renders them. Source clips may overlap between different narratives.
            </div>
            {settingsError && <p className="auto-story-error">{settingsError}</p>}
            {settings && !readyForStart && (
              <div className="auto-story-warning">
                Before starting, add the selected AI provider key and Hugging Face token in{' '}
                <Link to="/settings">Studio Settings</Link> and accept the{' '}
                <a href="https://huggingface.co/pyannote/speaker-diarization-3.1" target="_blank" rel="noopener noreferrer">
                  PyAnnote model terms
                </a>.
              </div>
            )}
          </div>

          {error && <div className="auto-story-error" role="alert">{error}</div>}
          <button
            type="submit"
            className="btn btn-primary"
            disabled={creating || uploading || !uploadFilename || !readyForStart}
          >
            {creating ? <><span className="spinner"></span> Starting…</> : '✨ Find storylines'}
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
  const active = !TERMINAL_STATUSES.includes(job.status)
  const progress = job.progress?.percent || 0

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
            Long episodes can take a while to transcribe. Keep the backend terminal open; you can return to this page later.
          </p>
          <button type="button" className="btn btn-danger btn-sm" onClick={requestCancel} disabled={cancelling}>
            {cancelling ? 'Requesting…' : 'Cancel'}
          </button>
        </div>
      )}

      {error && <div className="auto-story-error" role="alert">{error}</div>}
      {job.error && <div className="auto-story-error" role="alert">{job.error}</div>}

      {job.status === 'awaiting_storyline' && (
        <section className="auto-story-section">
          <h3>Choose which stories to create</h3>
          <p className="auto-story-help">
            Select one or more storylines. All are selected initially; uncheck any you do not want.
            After you continue, the backend automatically creates and renders the selected stories.
          </p>
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
          {state.storylines?.length === 0 && (
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
              {rediscovering ? <><span className="spinner"></span> Finding new storylines…</> : '🔄 Rerun story discovery'}
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={createSelectedStories}
              disabled={!selectedStorylineIds.length || selecting || rediscovering}
            >
              {selecting
                ? <><span className="spinner"></span> Creating stories…</>
                : `✅ Create & render ${selectedStorylineIds.length} ${selectedStorylineIds.length === 1 ? 'story' : 'stories'}`}
            </button>
          </div>
        </section>
      )}

      {job.status === 'awaiting_recipe' && !state.stories?.length && legacyRecipe && (
        <section className="auto-story-section">
          <h3>Review the dialogue beats</h3>
          <p className="auto-story-help">
            This job uses the earlier single-story workflow. Its schedule was prechecked; timing changes need a new validation.
          </p>
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
          <p className="auto-story-help">
            All {stories.length} narrative arcs and dialogue schedules have been created and checked before rendering.
            Review them together. Reconstruct any one story without changing the others, or adjust word-boundary timings.
          </p>
          <div className="auto-story-plan-list">
            {stories.map((story, storyIndex) => (
              <article className="card auto-story-storyline auto-story-plan" key={story.story_id}>
                <div className="auto-story-plan-heading">
                  <div>
                    <span className="auto-story-kicker">
                      Story {storyIndex + 1} of {stories.length} · {story.validation?.estimated_runtime_sec?.toFixed(1) || '—'}s planned
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
    </div>
  )
}

export default AutoStory
