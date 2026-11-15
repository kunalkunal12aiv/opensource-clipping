import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  cancelAutoStory,
  createAutoStory,
  fetchAutoStory,
  fetchSettings,
  renderAutoStory,
  selectAutoStoryline,
  uploadVideo,
  validateAutoStoryRecipe,
} from '../api'

const TERMINAL_STATUSES = ['completed', 'failed', 'cancelled']

function AutoStory() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const fileRef = useRef(null)
  const recipeTouched = useRef(false)
  const [settings, setSettings] = useState(null)
  const [settingsError, setSettingsError] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploadFilename, setUploadFilename] = useState('')
  const [uploading, setUploading] = useState(false)
  const [targetDuration, setTargetDuration] = useState(60)
  const [provider, setProvider] = useState('gemini')
  const [creating, setCreating] = useState(false)
  const [job, setJob] = useState(null)
  const [recipe, setRecipe] = useState(null)
  const [validated, setValidated] = useState(false)
  const [validating, setValidating] = useState(false)
  const [selecting, setSelecting] = useState(false)
  const [rendering, setRendering] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [error, setError] = useState('')

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
        if (data.auto_story?.recipe && !recipeTouched.current) {
          setRecipe(data.auto_story.recipe)
          setValidated(Boolean(data.auto_story.validation))
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
      setJob(await selectAutoStoryline(jobId, storylineId))
      recipeTouched.current = false
      setRecipe(null)
      setValidated(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setSelecting(false)
    }
  }

  const editBeat = (index, key, value) => {
    recipeTouched.current = true
    setValidated(false)
    setError('')
    setRecipe(current => ({
      ...current,
      beats: current.beats.map((beat, beatIndex) =>
        beatIndex === index ? { ...beat, [key]: value } : beat,
      ),
    }))
  }

  const validateRecipe = async () => {
    setValidating(true)
    setError('')
    try {
      const updated = await validateAutoStoryRecipe(jobId, recipe)
      setJob(updated)
      setRecipe(updated.auto_story.recipe)
      recipeTouched.current = false
      setValidated(true)
    } catch (err) {
      setError(err.message)
      setValidated(false)
    } finally {
      setValidating(false)
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
            <h2>Auto Story-Recipe</h2>
            <p>Create a reviewed, dialogue-led montage from one episode.</p>
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
            <h3 className="card-title">2. Set your montage length</h3>
            <div className="auto-story-options">
              <div className="form-group">
                <label className="form-label" htmlFor="auto-story-duration">Target duration</label>
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
              Transcription uses the local CPU with Whisper small. Auto Story makes two AI requests
              (storylines and dialogue beats); results are cached and invalid results are not retried automatically.
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
          <p>{state.source_name || job.upload_filename} · About {state.target_duration_sec || 60} seconds</p>
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
          <h3>3. Choose the storyline</h3>
          <p className="auto-story-help">Pick the arc you want the montage to tell. The AI will then select dialogue beats from that storyline.</p>
          <div className="auto-story-storylines">
            {state.storylines?.map((storyline, index) => (
              <article className="card auto-story-storyline" key={storyline.id}>
                <span className="auto-story-kicker">Storyline {index + 1}</span>
                <h4>{storyline.title}</h4>
                <p>{storyline.summary}</p>
                <div className="auto-story-meta">
                  <span>Characters: {storyline.key_characters?.join(', ') || 'Not specified'}</span>
                  <span>Speakers: {storyline.speaker_ids?.join(', ') || 'Not identified'}</span>
                  <span>Episode span: {Math.floor(storyline.start_sec / 60)}:{String(Math.floor(storyline.start_sec % 60)).padStart(2, '0')} – {Math.floor(storyline.end_sec / 60)}:{String(Math.floor(storyline.end_sec % 60)).padStart(2, '0')}</span>
                </div>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => chooseStoryline(storyline.id)}
                  disabled={selecting}
                >
                  {selecting ? 'Selecting beats…' : 'Build this storyline'}
                </button>
              </article>
            ))}
          </div>
        </section>
      )}

      {job.status === 'awaiting_recipe' && recipe && (
        <section className="auto-story-section">
          <h3>4. Review the dialogue beats</h3>
          <p className="auto-story-help">
            Adjust the start/end times if needed. Times snap to transcript word boundaries; validation checks pacing and total length before rendering.
          </p>
          <div className="auto-story-beats">
            {recipe.beats?.map((beat, index) => (
              <article className="card auto-story-beat" key={beat.beat_id || index}>
                <div className="auto-story-beat-heading">
                  <div>
                    <span className="auto-story-kicker">Beat {index + 1} · {beat.narrative_role || 'Dialogue'}</span>
                    <p className="auto-story-dialogue">“{beat.text}”</p>
                    <span className="auto-story-meta-text">
                      Speaker: {beat.speaker_ids?.join(', ') || 'Not identified'}
                    </span>
                  </div>
                </div>
                <div className="auto-story-time-fields">
                  <label>
                    Start (seconds)
                    <input
                      className="form-input"
                      type="number"
                      step="0.01"
                      value={beat.dialogue_start_sec}
                      onChange={event => editBeat(index, 'dialogue_start_sec', event.target.value)}
                    />
                  </label>
                  <label>
                    End (seconds)
                    <input
                      className="form-input"
                      type="number"
                      step="0.01"
                      value={beat.dialogue_end_sec}
                      onChange={event => editBeat(index, 'dialogue_end_sec', event.target.value)}
                    />
                  </label>
                </div>
              </article>
            ))}
          </div>
          {state.validation && validated && (
            <div className="auto-story-success">
              Recipe valid: {state.validation.beat_count} beats, estimated {state.validation.estimated_runtime_sec.toFixed(1)} seconds.
            </div>
          )}
          <div className="auto-story-actions">
            <button type="button" className="btn btn-secondary" onClick={validateRecipe} disabled={validating}>
              {validating ? <><span className="spinner"></span> Checking…</> : 'Check my recipe'}
            </button>
            <button type="button" className="btn btn-primary" onClick={approveRender} disabled={!validated || rendering}>
              {rendering ? <><span className="spinner"></span> Starting render…</> : '✅ Approve & render'}
            </button>
          </div>
        </section>
      )}

      {job.status === 'completed' && (
        <section className="auto-story-section">
          <h3>Your montage is ready</h3>
          {state.output_files?.some(file => file.filename.endsWith('.mp4')) && (
            <video
              className="auto-story-video"
              controls
              preload="metadata"
              src={state.output_files.find(file => file.filename.endsWith('.mp4')).download_url}
            />
          )}
          <div className="auto-story-downloads">
            {state.output_files?.map(file => (
              <a className="btn btn-secondary" href={file.download_url} download key={file.filename}>
                ⬇️ {file.filename}
              </a>
            ))}
          </div>
        </section>
      )}

      {job.status === 'failed' && (
        <section className="auto-story-section">
          <Link to="/auto-story" className="btn btn-primary">Start another Auto Story</Link>
        </section>
      )}
    </div>
  )
}

export default AutoStory
