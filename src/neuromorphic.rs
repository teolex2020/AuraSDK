//! Neuromorphic Export Module
//!
//! Exports SDR representations to hardware-accelerated formats:
//! - SpiNNaker (ARM-based neuromorphic platform)
//! - FPGA bitstream helpers
//! - Intel Loihi 2 compatibility
//!
//! # SDR to Spike Train Conversion
//! Active bits in SDR → spike events at specific timesteps
//!
//! # Use Cases
//! - Hardware-accelerated similarity search
//! - Ultra-low power edge inference
//! - Real-time pattern matching

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// SpiNNaker neuron population configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiNNakerPopulation {
    /// Population label/name
    pub label: String,
    /// Number of neurons
    pub n_neurons: u32,
    /// Neuron model (e.g., "IF_curr_exp")
    pub model: String,
    /// Model parameters
    pub params: HashMap<String, f64>,
}

impl Default for SpiNNakerPopulation {
    fn default() -> Self {
        let mut params = HashMap::new();
        params.insert("tau_m".to_string(), 20.0);
        params.insert("tau_syn_E".to_string(), 5.0);
        params.insert("tau_syn_I".to_string(), 5.0);
        params.insert("v_rest".to_string(), -65.0);
        params.insert("v_reset".to_string(), -65.0);
        params.insert("v_thresh".to_string(), -50.0);
        params.insert("tau_refrac".to_string(), 2.0);

        Self {
            label: "aura_sdr".to_string(),
            n_neurons: 262144, // Default SDR size
            model: "IF_curr_exp".to_string(),
            params,
        }
    }
}

/// Spike event for neuromorphic processing
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpikeEvent {
    /// Neuron/bit index
    pub neuron_id: u32,
    /// Timestamp in milliseconds
    pub timestamp_ms: f64,
}

/// SDR as spike train
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeTrain {
    /// Memory ID this spike train represents
    pub memory_id: String,
    /// Individual spike events
    pub spikes: Vec<SpikeEvent>,
    /// Duration of the pattern in ms
    pub duration_ms: f64,
    /// Number of active neurons
    pub active_count: usize,
}

/// SpiNNaker export configuration
#[derive(Debug, Clone)]
pub struct SpiNNakerExporter {
    /// Base time step in ms
    pub timestep_ms: f64,
    /// Pattern duration in ms
    pub pattern_duration_ms: f64,
    /// Inter-pattern gap in ms
    pub gap_ms: f64,
    /// Population configuration
    pub population: SpiNNakerPopulation,
}

impl Default for SpiNNakerExporter {
    fn default() -> Self {
        Self {
            timestep_ms: 1.0,
            pattern_duration_ms: 10.0,
            gap_ms: 5.0,
            population: SpiNNakerPopulation::default(),
        }
    }
}

impl SpiNNakerExporter {
    /// Create exporter with custom SDR size
    pub fn with_sdr_size(n_bits: u32) -> Self {
        let mut exporter = Self::default();
        exporter.population.n_neurons = n_bits;
        exporter
    }

    /// Convert SDR active bits to spike train
    pub fn sdr_to_spikes(&self, memory_id: &str, active_bits: &[u32]) -> SpikeTrain {
        let spikes: Vec<SpikeEvent> = active_bits
            .iter()
            .enumerate()
            .map(|(i, &neuron_id)| SpikeEvent {
                neuron_id,
                // Distribute spikes across pattern duration
                timestamp_ms: (i as f64 / active_bits.len() as f64) * self.pattern_duration_ms,
            })
            .collect();

        SpikeTrain {
            memory_id: memory_id.to_string(),
            spikes,
            duration_ms: self.pattern_duration_ms,
            active_count: active_bits.len(),
        }
    }

    /// Export multiple SDRs as concatenated spike trains
    pub fn export_spike_trains(
        &self,
        memories: &[(&str, &[u32])], // (id, active_bits)
    ) -> Vec<SpikeTrain> {
        let mut offset_ms = 0.0;
        let mut trains = Vec::new();

        for (id, bits) in memories {
            let mut train = self.sdr_to_spikes(id, bits);

            // Offset timestamps
            for spike in &mut train.spikes {
                spike.timestamp_ms += offset_ms;
            }

            offset_ms += self.pattern_duration_ms + self.gap_ms;
            trains.push(train);
        }

        trains
    }

    /// Generate PyNN-compatible Python script for SpiNNaker
    pub fn generate_pynn_script(&self, trains: &[SpikeTrain]) -> String {
        let mut script = String::new();

        // Header
        script.push_str("# Auto-generated Aura Memory → SpiNNaker export\n");
        script.push_str("# PyNN script for neuromorphic pattern matching\n\n");
        script.push_str("import spynnaker8 as sim\n");
        script.push_str("from pyNN.utility.plotting import Figure, Panel\n\n");

        // Setup
        script.push_str(&format!("sim.setup(timestep={})\n\n", self.timestep_ms));

        // Population
        script.push_str(&format!(
            "# SDR population ({} neurons)\n",
            self.population.n_neurons
        ));
        script.push_str(&format!(
            "sdr_pop = sim.Population({}, sim.{}(**{{\n",
            self.population.n_neurons, self.population.model
        ));
        for (key, value) in &self.population.params {
            script.push_str(&format!("    '{}': {},\n", key, value));
        }
        script.push_str("}), label='aura_sdr')\n\n");

        // Spike sources for each memory
        script.push_str("# Spike sources for each memory pattern\n");
        for (i, train) in trains.iter().enumerate() {
            script.push_str(&format!(
                "# Memory: {} ({} spikes)\n",
                train.memory_id, train.active_count
            ));

            // Create spike times dict
            script.push_str(&format!("spikes_{} = {{\n", i));
            let mut neuron_spikes: HashMap<u32, Vec<f64>> = HashMap::new();
            for spike in &train.spikes {
                neuron_spikes
                    .entry(spike.neuron_id)
                    .or_default()
                    .push(spike.timestamp_ms);
            }
            for (neuron, times) in &neuron_spikes {
                script.push_str(&format!("    {}: {:?},\n", neuron, times));
            }
            script.push_str("}\n\n");
        }

        // Record and run
        let total_time: f64 =
            trains.iter().map(|t| t.duration_ms).sum::<f64>() + (trains.len() as f64 * self.gap_ms);

        script.push_str("sdr_pop.record(['spikes', 'v'])\n");
        script.push_str(&format!("sim.run({})\n\n", total_time));

        // Extract data
        script.push_str("# Extract spike data\n");
        script.push_str("data = sdr_pop.get_data()\n");
        script.push_str("sim.end()\n\n");

        script.push_str("print(f'Recorded {len(data.segments[0].spiketrains)} spike trains')\n");

        script
    }

    /// Export to SpiNNaker-compatible binary format
    pub fn export_binary<P: AsRef<Path>>(
        &self,
        trains: &[SpikeTrain],
        output_path: P,
    ) -> Result<()> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        // Header: magic, version, n_patterns, n_neurons
        writer.write_all(b"AURA")?; // Magic
        writer.write_all(&[1u8])?; // Version
        writer.write_all(&(trains.len() as u32).to_le_bytes())?;
        writer.write_all(&self.population.n_neurons.to_le_bytes())?;

        // Each pattern
        for train in trains {
            // Pattern header: n_spikes, duration_ms
            writer.write_all(&(train.spikes.len() as u32).to_le_bytes())?;
            writer.write_all(&train.duration_ms.to_le_bytes())?;

            // Spikes: neuron_id (u32) + timestamp (f64)
            for spike in &train.spikes {
                writer.write_all(&spike.neuron_id.to_le_bytes())?;
                writer.write_all(&spike.timestamp_ms.to_le_bytes())?;
            }
        }

        writer.flush()?;
        Ok(())
    }
}

/// FPGA SDR representation (packed bits)
#[derive(Debug, Clone)]
pub struct FpgaSdr {
    /// SDR as packed bytes (bit vector)
    pub packed_bits: Vec<u8>,
    /// Total number of bits
    pub n_bits: u32,
    /// Number of active bits
    pub active_count: u32,
}

/// FPGA exporter for hardware acceleration
pub struct FpgaExporter {
    /// SDR width in bits
    pub sdr_width: u32,
    /// Memory depth (number of patterns)
    pub memory_depth: u32,
}

impl FpgaExporter {
    /// Create FPGA exporter with SDR configuration
    pub fn new(sdr_width: u32, memory_depth: u32) -> Self {
        Self {
            sdr_width,
            memory_depth,
        }
    }

    /// Pack active bits into byte array
    pub fn pack_sdr(&self, active_bits: &[u32]) -> FpgaSdr {
        let n_bytes = self.sdr_width.div_ceil(8) as usize;
        let mut packed = vec![0u8; n_bytes];

        for &bit in active_bits {
            if bit < self.sdr_width {
                let byte_idx = (bit / 8) as usize;
                let bit_idx = bit % 8;
                packed[byte_idx] |= 1 << bit_idx;
            }
        }

        FpgaSdr {
            packed_bits: packed,
            n_bits: self.sdr_width,
            active_count: active_bits.len() as u32,
        }
    }

    /// Generate Verilog module for SDR comparison
    pub fn generate_verilog_comparator(&self) -> String {
        let mut verilog = String::new();

        verilog.push_str("// Auto-generated Aura Memory SDR Comparator\n");
        verilog.push_str("// Computes Tanimoto similarity between two SDRs\n\n");

        verilog.push_str(&format!(
            "module aura_sdr_compare #(\n    parameter SDR_WIDTH = {}\n)(\n",
            self.sdr_width
        ));

        verilog.push_str("    input wire clk,\n");
        verilog.push_str("    input wire rst_n,\n");
        verilog.push_str("    input wire start,\n");
        verilog.push_str("    input wire [SDR_WIDTH-1:0] sdr_a,\n");
        verilog.push_str("    input wire [SDR_WIDTH-1:0] sdr_b,\n");
        verilog.push_str("    output reg [15:0] similarity,  // Fixed-point 0.16\n");
        verilog.push_str("    output reg done\n");
        verilog.push_str(");\n\n");

        // Popcount function
        verilog.push_str("    // Popcount (number of 1s)\n");
        verilog.push_str("    function automatic [15:0] popcount;\n");
        verilog.push_str("        input [SDR_WIDTH-1:0] bits;\n");
        verilog.push_str("        integer i;\n");
        verilog.push_str("        begin\n");
        verilog.push_str("            popcount = 0;\n");
        verilog.push_str("            for (i = 0; i < SDR_WIDTH; i = i + 1)\n");
        verilog.push_str("                popcount = popcount + bits[i];\n");
        verilog.push_str("        end\n");
        verilog.push_str("    endfunction\n\n");

        // Registers
        verilog.push_str("    reg [15:0] intersection;\n");
        verilog.push_str("    reg [15:0] union_count;\n");
        verilog.push_str("    reg [1:0] state;\n\n");

        verilog.push_str("    localparam IDLE = 2'b00;\n");
        verilog.push_str("    localparam COMPUTE = 2'b01;\n");
        verilog.push_str("    localparam DIVIDE = 2'b10;\n");
        verilog.push_str("    localparam DONE = 2'b11;\n\n");

        // State machine
        verilog.push_str("    always @(posedge clk or negedge rst_n) begin\n");
        verilog.push_str("        if (!rst_n) begin\n");
        verilog.push_str("            state <= IDLE;\n");
        verilog.push_str("            done <= 0;\n");
        verilog.push_str("            similarity <= 0;\n");
        verilog.push_str("        end else begin\n");
        verilog.push_str("            case (state)\n");
        verilog.push_str("                IDLE: begin\n");
        verilog.push_str("                    done <= 0;\n");
        verilog.push_str("                    if (start) state <= COMPUTE;\n");
        verilog.push_str("                end\n");
        verilog.push_str("                COMPUTE: begin\n");
        verilog.push_str("                    intersection <= popcount(sdr_a & sdr_b);\n");
        verilog.push_str("                    union_count <= popcount(sdr_a | sdr_b);\n");
        verilog.push_str("                    state <= DIVIDE;\n");
        verilog.push_str("                end\n");
        verilog.push_str("                DIVIDE: begin\n");
        verilog.push_str("                    // Tanimoto = intersection / union\n");
        verilog.push_str("                    if (union_count > 0)\n");
        verilog.push_str(
            "                        similarity <= (intersection << 16) / union_count;\n",
        );
        verilog.push_str("                    else\n");
        verilog.push_str("                        similarity <= 0;\n");
        verilog.push_str("                    state <= DONE;\n");
        verilog.push_str("                end\n");
        verilog.push_str("                DONE: begin\n");
        verilog.push_str("                    done <= 1;\n");
        verilog.push_str("                    state <= IDLE;\n");
        verilog.push_str("                end\n");
        verilog.push_str("            endcase\n");
        verilog.push_str("        end\n");
        verilog.push_str("    end\n\n");

        verilog.push_str("endmodule\n");

        verilog
    }

    /// Generate memory initialization file (MIF format)
    pub fn generate_mif(&self, patterns: &[FpgaSdr]) -> String {
        let mut mif = String::new();

        mif.push_str("-- Aura Memory SDR patterns\n");
        mif.push_str(&format!("DEPTH = {};\n", patterns.len()));
        mif.push_str(&format!("WIDTH = {};\n", self.sdr_width));
        mif.push_str("ADDRESS_RADIX = DEC;\n");
        mif.push_str("DATA_RADIX = BIN;\n");
        mif.push_str("CONTENT\n");
        mif.push_str("BEGIN\n");

        for (addr, pattern) in patterns.iter().enumerate() {
            // Convert to binary string
            let mut bits = String::new();
            for i in (0..self.sdr_width).rev() {
                let byte_idx = (i / 8) as usize;
                let bit_idx = i % 8;
                if byte_idx < pattern.packed_bits.len() {
                    let bit = (pattern.packed_bits[byte_idx] >> bit_idx) & 1;
                    bits.push(if bit == 1 { '1' } else { '0' });
                } else {
                    bits.push('0');
                }
            }
            mif.push_str(&format!("    {} : {};\n", addr, bits));
        }

        mif.push_str("END;\n");
        mif
    }

    /// Export patterns to binary file for FPGA loading
    pub fn export_binary<P: AsRef<Path>>(
        &self,
        patterns: &[FpgaSdr],
        output_path: P,
    ) -> Result<()> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        // Header
        writer.write_all(b"AFPG")?; // Magic: Aura FPGA
        writer.write_all(&self.sdr_width.to_le_bytes())?;
        writer.write_all(&(patterns.len() as u32).to_le_bytes())?;

        // Patterns
        for pattern in patterns {
            writer.write_all(&pattern.packed_bits)?;
        }

        writer.flush()?;
        Ok(())
    }
}

/// Intel Loihi 2 compatibility layer
pub struct Loihi2Exporter {
    /// Number of cores to use
    pub n_cores: u32,
    /// Neurons per core
    pub neurons_per_core: u32,
}

impl Loihi2Exporter {
    /// Create Loihi 2 exporter
    pub fn new(n_cores: u32) -> Self {
        Self {
            n_cores,
            neurons_per_core: 128, // Loihi 2 default
        }
    }

    /// Map SDR bits to Loihi cores
    pub fn map_to_cores(&self, active_bits: &[u32]) -> HashMap<u32, Vec<u32>> {
        let mut core_map: HashMap<u32, Vec<u32>> = HashMap::new();

        for &bit in active_bits {
            let core_id = bit / self.neurons_per_core;
            let local_neuron = bit % self.neurons_per_core;

            if core_id < self.n_cores {
                core_map.entry(core_id).or_default().push(local_neuron);
            }
        }

        core_map
    }

    /// Generate NxSDK-compatible Python code
    pub fn generate_nxsdk_code(&self, memory_id: &str, active_bits: &[u32]) -> String {
        let mut code = String::new();

        code.push_str("# Auto-generated Aura Memory → Loihi 2 export\n");
        code.push_str("# NxSDK code for neuromorphic pattern matching\n\n");
        code.push_str("import nxsdk.api.n2a as nx\n\n");

        code.push_str(&format!("# Memory pattern: {}\n", memory_id));
        code.push_str(&format!("# Active neurons: {}\n\n", active_bits.len()));

        // Create network
        code.push_str("net = nx.NxNet()\n\n");

        // Create compartment group
        code.push_str("# SDR compartment group\n");
        code.push_str(&format!(
            "sdr_cg = net.createCompartmentGroup(size={})\n\n",
            self.n_cores * self.neurons_per_core
        ));

        // Set active neurons
        code.push_str("# Active neurons in SDR\n");
        code.push_str(&format!("active_neurons = {:?}\n\n", active_bits));

        // Spike generator
        code.push_str("# Create spike generator for input\n");
        code.push_str("spike_gen = net.createSpikeGenProcess(numPorts=len(active_neurons))\n");
        code.push_str("for i, neuron_id in enumerate(active_neurons):\n");
        code.push_str("    spike_gen.addSpikes(i, [1])  # Spike at t=1\n\n");

        code.push_str("# Compile and run\n");
        code.push_str("compiler = nx.N2Compiler()\n");
        code.push_str("board = compiler.compile(net)\n");
        code.push_str("board.run(100)  # Run for 100 timesteps\n");
        code.push_str("board.disconnect()\n");

        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdr_to_spikes() {
        let exporter = SpiNNakerExporter::default();
        let active_bits = vec![100, 500, 1000, 2000];

        let train = exporter.sdr_to_spikes("test_memory", &active_bits);

        assert_eq!(train.memory_id, "test_memory");
        assert_eq!(train.active_count, 4);
        assert_eq!(train.spikes.len(), 4);

        // Spikes should be distributed across pattern duration
        assert!(train.spikes[0].timestamp_ms < train.spikes[3].timestamp_ms);
    }

    #[test]
    fn test_export_spike_trains() {
        let exporter = SpiNNakerExporter::default();
        let mem1_bits = vec![1u32, 2, 3];
        let mem2_bits = vec![10u32, 20, 30];
        let memories: Vec<(&str, &[u32])> = vec![("mem1", &mem1_bits), ("mem2", &mem2_bits)];

        let trains = exporter.export_spike_trains(&memories);

        assert_eq!(trains.len(), 2);

        // Second train should have offset timestamps
        let first_end = trains[0].duration_ms;
        let second_start = trains[1].spikes[0].timestamp_ms;
        assert!(second_start > first_end);
    }

    #[test]
    fn test_fpga_pack_sdr() {
        let exporter = FpgaExporter::new(256, 100);
        let active_bits = vec![0, 7, 8, 15, 255];

        let packed = exporter.pack_sdr(&active_bits);

        assert_eq!(packed.n_bits, 256);
        assert_eq!(packed.active_count, 5);
        assert_eq!(packed.packed_bits.len(), 32); // 256/8 bytes

        // Check specific bits
        assert_eq!(packed.packed_bits[0] & 0x01, 1); // bit 0
        assert_eq!(packed.packed_bits[0] & 0x80, 128); // bit 7
        assert_eq!(packed.packed_bits[1] & 0x01, 1); // bit 8
        assert_eq!(packed.packed_bits[31] & 0x80, 128); // bit 255
    }

    #[test]
    fn test_loihi_core_mapping() {
        let exporter = Loihi2Exporter::new(16);
        let active_bits = vec![0, 127, 128, 256, 1000];

        let core_map = exporter.map_to_cores(&active_bits);

        // 0, 127 -> core 0
        // 128 -> core 1
        // 256 -> core 2
        // 1000 -> core 7
        assert!(core_map.contains_key(&0));
        assert!(core_map.contains_key(&1));
        assert!(core_map.contains_key(&2));
        assert!(core_map.contains_key(&7));

        assert_eq!(core_map.get(&0).unwrap().len(), 2); // neurons 0 and 127
    }

    #[test]
    fn test_generate_verilog() {
        let exporter = FpgaExporter::new(512, 100);
        let verilog = exporter.generate_verilog_comparator();

        assert!(verilog.contains("module aura_sdr_compare"));
        assert!(verilog.contains("SDR_WIDTH = 512"));
        assert!(verilog.contains("popcount"));
        assert!(verilog.contains("Tanimoto"));
    }

    #[test]
    fn test_generate_mif() {
        let exporter = FpgaExporter::new(16, 4);
        let patterns = vec![exporter.pack_sdr(&[0, 8]), exporter.pack_sdr(&[1, 9])];

        let mif = exporter.generate_mif(&patterns);

        assert!(mif.contains("DEPTH = 2"));
        assert!(mif.contains("WIDTH = 16"));
        assert!(mif.contains("CONTENT"));
    }

    #[test]
    fn test_generate_pynn() {
        let exporter = SpiNNakerExporter::with_sdr_size(1024);
        let train = exporter.sdr_to_spikes("test", &[10, 20, 30]);
        let script = exporter.generate_pynn_script(&[train]);

        assert!(script.contains("spynnaker8"));
        assert!(script.contains("Population(1024"));
        assert!(script.contains("sim.run"));
    }
}
