import express from 'express'
import dotenv from 'dotenv'
import OpenAI from 'openai'

dotenv.config()

const app = express()
const port = process.env.PORT || 8787

app.use(express.json())

const apiKey = process.env.OPENAI_API_KEY
if (!apiKey) {
  console.warn('OPENAI_API_KEY is not set. Requests will fail until it is configured in your .env')
}

const client = apiKey ? new OpenAI({ apiKey }) : null
const defaultModel = process.env.OPENAI_MODEL || 'gpt-4o-mini'

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, hasKey: Boolean(apiKey) })
})

app.post('/api/ask', async (req, res) => {
  try {
    if (!client) return res.status(500).json({ error: 'OPENAI_API_KEY not configured on server' })
    const { prompt, model } = req.body || {}
    if (!prompt || typeof prompt !== 'string') {
      return res.status(400).json({ error: 'prompt (string) is required' })
    }

    const completion = await client.chat.completions.create({
      model: model || defaultModel,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.7,
    })

    const text = completion.choices?.[0]?.message?.content ?? ''
    res.json({ text })
  } catch (err) {
    // Attempt to surface OpenAI API error details to the client
    const anyErr = err
    const status = anyErr?.status || 500
    const message = anyErr?.error?.message || anyErr?.message || 'Unknown error'
    console.error('OpenAI error:', anyErr)
    res.status(status).json({ error: message })
  }
})

app.listen(port, () => {
  console.log(`API server listening at http://localhost:${port}`)
})

