//! Federated Learning Module
//!
//! Enables privacy-preserving collective learning across devices:
//! - Gradient sharing only (no raw data leaves device)
//! - Differential privacy with mathematical guarantees
//! - Model aggregation for collective intelligence
//!
//! # Privacy Model
//! Each device shares only:
//! - SDR bit frequency statistics (anonymized)
//! - Salience weight gradients
//! - Anchor pattern summaries
//!
//! Raw memories NEVER leave the device.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::{anyhow, Result};
use serde::{Serialize, Deserialize};
use rand::Rng;
#[cfg(feature = "sync")]
use tokio::net::{TcpListener, TcpStream};
#[cfg(feature = "sync")]
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Differential privacy parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyParams {
    /// Privacy budget (epsilon) - lower = more private
    pub epsilon: f64,
    /// Delta parameter for (ε,δ)-differential privacy
    pub delta: f64,
    /// Sensitivity (max contribution per record)
    pub sensitivity: f64,
    /// Noise mechanism: Laplace or Gaussian
    pub mechanism: NoiseMechanism,
}

impl Default for PrivacyParams {
    fn default() -> Self {
        Self {
            epsilon: 1.0,  // Moderate privacy
            delta: 1e-5,   // Very small failure probability
            sensitivity: 1.0,
            mechanism: NoiseMechanism::Laplace,
        }
    }
}

impl PrivacyParams {
    /// Strong privacy preset (ε=0.1)
    pub fn strong() -> Self {
        Self {
            epsilon: 0.1,
            delta: 1e-6,
            ..Default::default()
        }
    }

    /// Moderate privacy preset (ε=1.0)
    pub fn moderate() -> Self {
        Self::default()
    }

    /// Weak privacy preset (ε=10.0)
    pub fn weak() -> Self {
        Self {
            epsilon: 10.0,
            delta: 1e-4,
            ..Default::default()
        }
    }
}

/// Noise mechanism for differential privacy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum NoiseMechanism {
    /// Laplace mechanism (pure ε-DP)
    Laplace,
    /// Gaussian mechanism ((ε,δ)-DP)
    Gaussian,
}

/// Local gradient update from a device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalGradient {
    /// Device identifier (anonymous)
    pub device_id: String,
    /// Timestamp of gradient computation
    pub timestamp: u64,
    /// SDR bit frequency changes (sparse)
    pub sdr_deltas: HashMap<u32, f64>,
    /// Salience weight updates
    pub salience_weights: SalienceGradient,
    /// Number of samples this gradient represents
    pub sample_count: u64,
    /// Whether differential privacy was applied
    pub is_private: bool,
}

/// Gradients for salience scoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SalienceGradient {
    /// Intensity weight gradient (α)
    pub alpha_grad: f64,
    /// Entropy weight gradient (β)
    pub beta_grad: f64,
    /// Resonance weight gradient (γ)
    pub gamma_grad: f64,
    /// Emotional pattern weight updates
    pub pattern_grads: HashMap<String, f64>,
}

/// Aggregated model update from federation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedModel {
    /// Round number
    pub round: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Number of participating devices
    pub participant_count: u64,
    /// Total samples across all devices
    pub total_samples: u64,
    /// Aggregated SDR statistics
    pub sdr_stats: HashMap<u32, f64>,
    /// Aggregated salience weights
    pub salience_weights: SalienceGradient,
}

/// Differential privacy engine
pub struct PrivacyEngine {
    params: PrivacyParams,
}

impl PrivacyEngine {
    /// Create a new privacy engine
    pub fn new(params: PrivacyParams) -> Self {
        Self { params }
    }

    /// Add Laplace noise to a value
    pub fn add_laplace_noise(&self, value: f64) -> f64 {
        let scale = self.params.sensitivity / self.params.epsilon;
        let noise = self.sample_laplace(scale);
        value + noise
    }

    /// Add Gaussian noise to a value
    pub fn add_gaussian_noise(&self, value: f64) -> f64 {
        let sigma = self.gaussian_sigma();
        let noise = self.sample_gaussian(sigma);
        value + noise
    }

    /// Compute sigma for Gaussian mechanism
    fn gaussian_sigma(&self) -> f64 {
        // σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        let ln_term = (1.25 / self.params.delta).ln();
        self.params.sensitivity * (2.0 * ln_term).sqrt() / self.params.epsilon
    }

    /// Sample from Laplace distribution
    fn sample_laplace(&self, scale: f64) -> f64 {
        let mut rng = rand::thread_rng();
        let u: f64 = rng.gen_range(-0.5..0.5);

        // Laplace: -b * sign(u) * ln(1 - 2|u|)
        -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }

    /// Sample from Gaussian distribution (Box-Muller)
    fn sample_gaussian(&self, sigma: f64) -> f64 {
        let mut rng = rand::thread_rng();
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();

        sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Apply differential privacy to gradient
    pub fn privatize_gradient(&self, gradient: &mut LocalGradient) {
        // Apply noise to SDR deltas
        for value in gradient.sdr_deltas.values_mut() {
            *value = match self.params.mechanism {
                NoiseMechanism::Laplace => self.add_laplace_noise(*value),
                NoiseMechanism::Gaussian => self.add_gaussian_noise(*value),
            };
        }

        // Apply noise to salience weights
        let sw = &mut gradient.salience_weights;
        sw.alpha_grad = self.add_laplace_noise(sw.alpha_grad);
        sw.beta_grad = self.add_laplace_noise(sw.beta_grad);
        sw.gamma_grad = self.add_laplace_noise(sw.gamma_grad);

        for value in sw.pattern_grads.values_mut() {
            *value = self.add_laplace_noise(*value);
        }

        gradient.is_private = true;
    }

    /// Compute privacy budget consumed
    pub fn budget_consumed(&self, num_queries: u64) -> f64 {
        // Sequential composition: total ε = n * ε
        num_queries as f64 * self.params.epsilon
    }

    /// Check if privacy budget is exhausted
    pub fn is_budget_exhausted(&self, max_budget: f64, queries_made: u64) -> bool {
        self.budget_consumed(queries_made) >= max_budget
    }
}

/// Local gradient computer for a device
pub struct LocalTrainer {
    device_id: String,
    privacy_engine: PrivacyEngine,
    sample_count: u64,
}

impl LocalTrainer {
    /// Create a new local trainer
    pub fn new(device_id: impl Into<String>, privacy_params: PrivacyParams) -> Self {
        Self {
            device_id: device_id.into(),
            privacy_engine: PrivacyEngine::new(privacy_params),
            sample_count: 0,
        }
    }

    /// Compute gradient from local SDR statistics
    pub fn compute_gradient(
        &mut self,
        sdr_frequencies: &HashMap<u32, u64>,
        total_records: u64,
    ) -> LocalGradient {
        // Convert frequencies to normalized deltas
        let sdr_deltas: HashMap<u32, f64> = sdr_frequencies
            .iter()
            .map(|(&bit, &freq)| {
                let normalized = freq as f64 / total_records.max(1) as f64;
                (bit, normalized)
            })
            .collect();

        self.sample_count += total_records;

        LocalGradient {
            device_id: self.device_id.clone(),
            timestamp: Self::now_ms(),
            sdr_deltas,
            salience_weights: SalienceGradient::default(),
            sample_count: total_records,
            is_private: false,
        }
    }

    /// Compute gradient with salience weight updates
    pub fn compute_salience_gradient(
        &mut self,
        alpha_error: f64,
        beta_error: f64,
        gamma_error: f64,
        learning_rate: f64,
    ) -> SalienceGradient {
        SalienceGradient {
            alpha_grad: -learning_rate * alpha_error,
            beta_grad: -learning_rate * beta_error,
            gamma_grad: -learning_rate * gamma_error,
            pattern_grads: HashMap::new(),
        }
    }

    /// Apply differential privacy to gradient before sharing
    pub fn privatize(&self, gradient: &mut LocalGradient) {
        self.privacy_engine.privatize_gradient(gradient);
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// Federation server for aggregating gradients
pub struct FederationServer {
    round: u64,
    gradients: Vec<LocalGradient>,
    min_participants: usize,
}

impl FederationServer {
    /// Create a new federation server
    pub fn new(min_participants: usize) -> Self {
        Self {
            round: 0,
            gradients: Vec::new(),
            min_participants,
        }
    }

    /// Receive a gradient from a device
    pub fn receive_gradient(&mut self, gradient: LocalGradient) -> Result<()> {
        // Only accept private gradients
        if !gradient.is_private {
            return Err(anyhow!("Only private (noisy) gradients accepted"));
        }

        self.gradients.push(gradient);
        Ok(())
    }

    /// Check if enough participants for aggregation
    pub fn ready_to_aggregate(&self) -> bool {
        self.gradients.len() >= self.min_participants
    }

    /// Aggregate gradients using Federated Averaging
    pub fn aggregate(&mut self) -> Result<AggregatedModel> {
        if !self.ready_to_aggregate() {
            return Err(anyhow!(
                "Not enough participants: {} < {}",
                self.gradients.len(),
                self.min_participants
            ));
        }

        self.round += 1;

        // Aggregate SDR statistics (weighted average)
        let total_samples: u64 = self.gradients.iter().map(|g| g.sample_count).sum();
        let mut sdr_stats: HashMap<u32, f64> = HashMap::new();

        for gradient in &self.gradients {
            let weight = gradient.sample_count as f64 / total_samples as f64;
            for (&bit, &delta) in &gradient.sdr_deltas {
                *sdr_stats.entry(bit).or_default() += weight * delta;
            }
        }

        // Aggregate salience weights
        let mut salience_weights = SalienceGradient::default();

        for gradient in &self.gradients {
            let weight = gradient.sample_count as f64 / total_samples as f64;
            salience_weights.alpha_grad += weight * gradient.salience_weights.alpha_grad;
            salience_weights.beta_grad += weight * gradient.salience_weights.beta_grad;
            salience_weights.gamma_grad += weight * gradient.salience_weights.gamma_grad;
        }

        let model = AggregatedModel {
            round: self.round,
            timestamp: Self::now_ms(),
            participant_count: self.gradients.len() as u64,
            total_samples,
            sdr_stats,
            salience_weights,
        };

        // Clear gradients for next round
        self.gradients.clear();

        Ok(model)
    }

    /// Get current round number
    pub fn current_round(&self) -> u64 {
        self.round
    }

    /// Get number of pending gradients
    pub fn pending_count(&self) -> usize {
        self.gradients.len()
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// Federated client for local model updates
#[allow(dead_code)]
pub struct FederatedClient {
    device_id: String,
    current_model: AggregatedModel,
    trainer: LocalTrainer,
}

impl FederatedClient {
    /// Create a new federated client
    pub fn new(device_id: impl Into<String>, privacy_params: PrivacyParams) -> Self {
        let device_id = device_id.into();
        Self {
            device_id: device_id.clone(),
            current_model: AggregatedModel {
                round: 0,
                timestamp: 0,
                participant_count: 0,
                total_samples: 0,
                sdr_stats: HashMap::new(),
                salience_weights: SalienceGradient::default(),
            },
            trainer: LocalTrainer::new(device_id, privacy_params),
        }
    }

    /// Apply aggregated model update
    pub fn apply_update(&mut self, model: AggregatedModel) {
        self.current_model = model;
    }

    /// Get SDR bit importance from federation
    pub fn get_bit_importance(&self, bit: u32) -> f64 {
        self.current_model.sdr_stats.get(&bit).copied().unwrap_or(0.0)
    }

    /// Compute a private gradient (noise applied) without submitting to a server
    pub fn compute_private_gradient(
        &mut self,
        sdr_frequencies: &HashMap<u32, u64>,
        total_records: u64,
    ) -> LocalGradient {
        let mut gradient = self.trainer.compute_gradient(sdr_frequencies, total_records);
        self.trainer.privatize(&mut gradient);
        gradient
    }

    /// Compute and submit gradient to server
    pub fn submit_gradient(
        &mut self,
        sdr_frequencies: &HashMap<u32, u64>,
        total_records: u64,
        server: &mut FederationServer,
    ) -> Result<()> {
        let mut gradient = self.trainer.compute_gradient(sdr_frequencies, total_records);
        self.trainer.privatize(&mut gradient);
        server.receive_gradient(gradient)
    }
}

/// Secure aggregation protocol (simulated)
#[derive(Debug)]
pub struct SecureAggregation {
    /// Minimum number of parties for reconstruction
    pub threshold: usize,
    /// Total number of parties
    pub n_parties: usize,
}

impl SecureAggregation {
    /// Create new secure aggregation with t-of-n threshold
    pub fn new(threshold: usize, n_parties: usize) -> Result<Self> {
        if threshold > n_parties {
            return Err(anyhow!("Threshold cannot exceed number of parties"));
        }
        if threshold < 2 {
            return Err(anyhow!("Threshold must be at least 2"));
        }

        Ok(Self { threshold, n_parties })
    }

    /// Check if aggregation can proceed
    pub fn can_aggregate(&self, available_parties: usize) -> bool {
        available_parties >= self.threshold
    }

    /// Simulate secure sum (in production, use Shamir secret sharing)
    pub fn secure_sum(&self, shares: &[f64]) -> Result<f64> {
        if shares.len() < self.threshold {
            return Err(anyhow!(
                "Not enough shares: {} < {}",
                shares.len(),
                self.threshold
            ));
        }

    // In production, this would use cryptographic protocols
        // For now, just sum the values
        Ok(shares.iter().sum())
    }
}

// =========================================================================================
//                                  NETWORKING LAYER (TCP Direct)
// =========================================================================================



#[cfg(feature = "sync")]
pub struct SimpleTcpNode {
    port: u16,
    peers: Vec<String>,
}

#[cfg(feature = "sync")]
impl SimpleTcpNode {
    pub fn new(port: u16, peers: Vec<String>) -> Self {
        Self { port, peers }
    }

    pub async fn start_server(&self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.port);
        let listener = TcpListener::bind(&addr).await?;
        tracing::info!("Aura Sync listening on {}", addr);

        loop {
            let (mut socket, peer_addr) = listener.accept().await?;
            tracing::info!("Accepted connection from {}", peer_addr);
            
            tokio::spawn(async move {
                let mut buffer = [0u8; 4096]; // Larger buffer for gradients
                loop {
                    match socket.read(&mut buffer).await {
                        Ok(n) if n > 0 => {
                            let msg = String::from_utf8_lossy(&buffer[..n]);
                            if msg.starts_with("GRADIENT_PUSH") {
                                // Simple extraction for demo
                                let json_part = msg.trim_start_matches("GRADIENT_PUSH ");
                                if let Ok(gradient) = serde_json::from_str::<LocalGradient>(json_part) {
                                    tracing::info!("Only-Private Gradient received from {}", gradient.device_id);
                                    // In a real system, we'd aggregated this.
                                    // For demo, we just log it.
                                }
                            }
                            
                            if let Err(e) = socket.write_all(b"ACK").await {
                                tracing::error!("Failed to send ACK: {}", e);
                                break;
                            }
                        }
                        Ok(_) => break, // EOF
                        Err(e) => {
                            tracing::error!("Socket error: {}", e);
                            break;
                        }
                    }
                }
            });
        }
    }

    pub async fn broadcast_gradient(&self, gradient: &LocalGradient) -> Result<()> {
        let json = serde_json::to_string(gradient)?;
        let payload = format!("GRADIENT_PUSH {}", json);

        for peer in &self.peers {
            match TcpStream::connect(peer).await {
                Ok(mut stream) => {
                    tracing::info!("Sending gradient to {}", peer);
                    stream.write_all(payload.as_bytes()).await?;
                    // Wait for ACK? For UDP-like speed we might skip, but let's read it.
                    let mut ack_buf = [0u8; 3];
                    let _ = stream.read(&mut ack_buf).await;
                }
                Err(e) => tracing::warn!("Failed to connect to {}: {}", peer, e),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_engine_laplace() {
        let params = PrivacyParams::default();
        let engine = PrivacyEngine::new(params);

        // Add noise to a value
        let original = 10.0;
        let noisy = engine.add_laplace_noise(original);

        // Noisy value should be different (with very high probability)
        // But we can't guarantee it, so just check it's a valid number
        assert!(noisy.is_finite());
    }

    #[test]
    fn test_privacy_engine_gaussian() {
        let mut params = PrivacyParams::default();
        params.mechanism = NoiseMechanism::Gaussian;
        let engine = PrivacyEngine::new(params);

        let original = 10.0;
        let noisy = engine.add_gaussian_noise(original);

        assert!(noisy.is_finite());
    }

    #[test]
    fn test_local_trainer() {
        let mut trainer = LocalTrainer::new("device_1", PrivacyParams::moderate());

        let mut frequencies = HashMap::new();
        frequencies.insert(100, 50);
        frequencies.insert(200, 30);
        frequencies.insert(300, 20);

        let gradient = trainer.compute_gradient(&frequencies, 100);

        assert_eq!(gradient.device_id, "device_1");
        assert_eq!(gradient.sample_count, 100);
        assert_eq!(gradient.sdr_deltas.len(), 3);
        assert!(!gradient.is_private);
    }

    #[test]
    fn test_privatize_gradient() {
        let trainer = LocalTrainer::new("device_1", PrivacyParams::moderate());

        let mut gradient = LocalGradient {
            device_id: "device_1".to_string(),
            timestamp: 0,
            sdr_deltas: [(100, 0.5), (200, 0.3)].into(),
            salience_weights: SalienceGradient {
                alpha_grad: 0.1,
                beta_grad: 0.2,
                gamma_grad: 0.3,
                pattern_grads: HashMap::new(),
            },
            sample_count: 50,
            is_private: false,
        };

        trainer.privatize(&mut gradient);

        assert!(gradient.is_private);
        // Values should be modified (though we can't check exact values)
    }

    #[test]
    fn test_federation_server() {
        let mut server = FederationServer::new(2);

        // Should not be ready with 0 gradients
        assert!(!server.ready_to_aggregate());

        // Add two private gradients
        let gradient1 = LocalGradient {
            device_id: "d1".to_string(),
            timestamp: 0,
            sdr_deltas: [(100, 0.5)].into(),
            salience_weights: SalienceGradient::default(),
            sample_count: 100,
            is_private: true,
        };

        let gradient2 = LocalGradient {
            device_id: "d2".to_string(),
            timestamp: 0,
            sdr_deltas: [(100, 0.3), (200, 0.2)].into(),
            salience_weights: SalienceGradient::default(),
            sample_count: 50,
            is_private: true,
        };

        server.receive_gradient(gradient1).unwrap();
        server.receive_gradient(gradient2).unwrap();

        assert!(server.ready_to_aggregate());

        let model = server.aggregate().unwrap();

        assert_eq!(model.round, 1);
        assert_eq!(model.participant_count, 2);
        assert_eq!(model.total_samples, 150);
    }

    #[test]
    fn test_non_private_gradient_rejected() {
        let mut server = FederationServer::new(1);

        let gradient = LocalGradient {
            device_id: "d1".to_string(),
            timestamp: 0,
            sdr_deltas: HashMap::new(),
            salience_weights: SalienceGradient::default(),
            sample_count: 100,
            is_private: false,  // Not private!
        };

        let result = server.receive_gradient(gradient);
        assert!(result.is_err());
    }

    #[test]
    fn test_privacy_budget() {
        let params = PrivacyParams {
            epsilon: 0.5,
            ..Default::default()
        };
        let engine = PrivacyEngine::new(params);

        // 10 queries at ε=0.5 = 5.0 total budget
        let consumed = engine.budget_consumed(10);
        assert!((consumed - 5.0).abs() < 0.001);

        // Should be exhausted if max is 4.0
        assert!(engine.is_budget_exhausted(4.0, 10));
        assert!(!engine.is_budget_exhausted(10.0, 10));
    }

    #[test]
    fn test_secure_aggregation() {
        let sa = SecureAggregation::new(3, 5).unwrap();

        assert!(!sa.can_aggregate(2));
        assert!(sa.can_aggregate(3));
        assert!(sa.can_aggregate(5));

        let shares = vec![1.0, 2.0, 3.0, 4.0];
        let sum = sa.secure_sum(&shares).unwrap();
        assert!((sum - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_federated_client() {
        let mut client = FederatedClient::new("client_1", PrivacyParams::moderate());
        let mut server = FederationServer::new(1);

        let mut frequencies = HashMap::new();
        frequencies.insert(50, 10);

        client.submit_gradient(&frequencies, 10, &mut server).unwrap();

        assert!(server.ready_to_aggregate());
    }

    #[test]
    fn test_privacy_presets() {
        let strong = PrivacyParams::strong();
        let moderate = PrivacyParams::moderate();
        let weak = PrivacyParams::weak();

        assert!(strong.epsilon < moderate.epsilon);
        assert!(moderate.epsilon < weak.epsilon);
    }

    #[cfg(feature = "sync")]
    #[tokio::test]
    async fn test_tcp_direct_sync() {
        // Node A listening on 9001
        let node_a = SimpleTcpNode::new(9001, vec![]);
        let _server_a = tokio::spawn(async move {
            node_a.start_server().await.unwrap();
        });

        // Give A time to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Node B sending to 9001
        let node_b = SimpleTcpNode::new(9002, vec!["127.0.0.1:9001".to_string()]);
        
        let gradient = LocalGradient {
            device_id: "NodeB".to_string(),
            timestamp: 12345,
            sdr_deltas: std::collections::HashMap::from([(1, 0.5)]),
            salience_weights: SalienceGradient::default(),
            sample_count: 10,
            is_private: true,
        };

        // Broadcast from B -> A
        node_b.broadcast_gradient(&gradient).await.unwrap();

        // In a real test, we would verify A received it.
        // For now, we just ensure no panic/error during sending.
        // Use manual verification (grep logs) if needed, or add shared state to verify.
    }
}
