import express from 'express';
import cors from 'cors';
import multer from 'multer';
import FormData from 'form-data';
import axios from 'axios';
import morgan from 'morgan';

const app = express();
const PORT = process.env.PORT || 5000;
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://127.0.0.1:8000';

// Multer memory storage for direct streaming to Python CV microservice
const upload = multer({ storage: multer.memoryStorage() });

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use(morgan('dev'));

// System Health Endpoint
app.get('/api/health', async (req, res) => {
  let pythonStatus = 'unreachable';
  let pythonFeatures = [];
  try {
    const pyRes = await axios.get(`${PYTHON_SERVICE_URL}/api/health`, { timeout: 3000 });
    pythonStatus = pyRes.data.status || 'healthy';
    pythonFeatures = pyRes.data.features || [];
  } catch (err) {
    pythonStatus = 'offline';
  }

  res.json({
    status: 'online',
    gateway: 'SpillTrace Node.js Orchestration Gateway',
    version: '1.0.0',
    python_service: {
      url: PYTHON_SERVICE_URL,
      status: pythonStatus,
      features: pythonFeatures
    }
  });
});

// Samples Catalog
app.get('/api/samples', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_SERVICE_URL}/api/samples`, { timeout: 5000 });
    res.json(response.data);
  } catch (error) {
    console.error('[Gateway] Error fetching samples:', error.message);
    res.status(502).json({ error: 'Failed to communicate with Python CV microservice' });
  }
});

// Detection Proxy
app.post('/api/detect', upload.single('file'), async (req, res) => {
  try {
    const formData = new FormData();

    if (req.body.sample_id) {
      formData.append('sample_id', req.body.sample_id);
    }

    if (req.file) {
      formData.append('file', req.file.buffer, {
        filename: req.file.originalname,
        contentType: req.file.mimetype,
      });
    }

    if (req.body.k_factor) formData.append('k_factor', req.body.k_factor);
    if (req.body.min_area_km2) formData.append('min_area_km2', req.body.min_area_km2);
    if (req.body.min_elongation) formData.append('min_elongation', req.body.min_elongation);

    const response = await axios.post(`${PYTHON_SERVICE_URL}/api/detect`, formData, {
      headers: formData.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 30000,
    });

    res.json(response.data);
  } catch (error) {
    console.error('[Gateway] Detection error:', error.message);
    const status = error.response ? error.response.status : 500;
    const detail = error.response ? error.response.data : { error: error.message };
    res.status(status).json(detail);
  }
});

// Benchmark Proxy
app.post('/api/benchmark', async (req, res) => {
  try {
    const response = await axios.post(`${PYTHON_SERVICE_URL}/api/benchmark`, req.body, { timeout: 10000 });
    res.json(response.data);
  } catch (error) {
    console.error('[Gateway] Benchmark error:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Future Phase Placeholders
app.get('/api/phase2/drift-preview', (req, res) => {
  res.json({
    status: 'ready_for_phase_2',
    description: 'Backward drift trajectory modeling using Open-Meteo ocean currents & wind vectors'
  });
});

app.get('/api/phase3/ais-preview', (req, res) => {
  res.json({
    status: 'ready_for_phase_3',
    description: 'Spatio-temporal AIS vessel corridor matching & attribution ranking'
  });
});

app.listen(PORT, () => {
  console.log(`[SpillTrace Gateway] Running on http://localhost:${PORT}`);
  console.log(`[SpillTrace Gateway] Proxying Python microservice at ${PYTHON_SERVICE_URL}`);
});
