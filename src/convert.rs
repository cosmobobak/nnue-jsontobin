use serde::Deserialize;

const INPUT_SIZE: usize = 768;

// example of the json format:
// {
//     "perspective.weight": [ ... ],
//     "perspective.bias": [ ... ],
//     "factoriser.weight": [ ... ],
//     "factoriser.bias": [ ... ],
//     "out.weight": [ ... ],
//     "out.bias": [ ... ]
// }
// the "factoriser" fields are optional, and will be ignored if they are not present.
// we also accept aliases for the field names, e.g. "ft.weight" instead of "perspective.weight",
// and "fft.weight" instead of "factoriser.weight".

#[derive(Deserialize)]
struct FullNetworkWeights {
    #[serde(rename = "perspective.weight")]
    #[serde(alias = "ft.weight")]
    perspective_weight: Box<[Box<[f64]>]>,
    #[serde(rename = "perspective.bias")]
    #[serde(alias = "ft.bias")]
    perspective_bias: Box<[f64]>,
    #[serde(rename = "factoriser.weight")]
    #[serde(alias = "fft.weight")]
    factoriser_weight: Option<Box<[Box<[f64]>]>>,
    #[serde(rename = "factoriser.bias")]
    #[serde(alias = "fft.bias")]
    factoriser_bias: Option<Box<[f64]>>,
    #[serde(rename = "psqt.weight")]
    psqt_weight: Option<Box<[Box<[f64]>]>>,
    #[serde(rename = "out.weight")]
    output_weight: Box<[Box<[f64]>]>,
    #[serde(rename = "out.bias")]
    output_bias: Box<[f64]>,
}

struct MergedNetworkWeights {
    pub perspective_weight: Box<[Box<[f64]>]>,
    pub perspective_bias: Box<[f64]>,
    pub output_weight: Box<[Box<[f64]>]>,
    pub output_bias: Box<[f64]>,
    pub psqt_weight: Option<Box<[f64]>>,
}

pub struct QuantisedMergedNetwork {
    pub feature_weights: Vec<i16>,
    pub feature_bias: Vec<i16>,
    pub output_weights: Vec<i16>,
    pub output_bias: Vec<i16>,
    pub psqt_weights: Option<Vec<i16>>,
    pub has_buckets: bool,
    pub hidden_size: usize,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum DoTranspose {
    Yes,
    No,
}

fn quantise_neurons(
    weights: &[Box<[f64]>],
    buffer: &mut [i16],
    stride: usize,
    k: i32,
    flip: DoTranspose,
) {
    #![allow(clippy::cast_possible_truncation)]
    for (i, output) in weights.iter().enumerate() {
        for (j, weight) in output.iter().enumerate() {
            let index = if flip == DoTranspose::Yes {
                j * stride + i
            } else {
                i * stride + j
            };
            buffer[index] = (weight * f64::from(k)) as i16;
        }
    }
}

fn quantise_biases(biases: &[f64], buffer: &mut [i16], k: i32) {
    #![allow(clippy::cast_possible_truncation)]
    for (i, bias) in biases.iter().enumerate() {
        buffer[i] = (bias * f64::from(k)) as i16;
    }
}

fn quantise_psqt(psqt: &[f64], buffer: &mut [i16], k: i32) {
    #![allow(clippy::cast_possible_truncation)]
    for (i, weight) in psqt.iter().enumerate() {
        buffer[i] = (weight * f64::from(k)) as i16;
    }
}

#[allow(clippy::too_many_lines)]
pub fn from_json(
    json: &str,
    qa: i32,
    qb: i32,
) -> Result<QuantisedMergedNetwork, Box<dyn std::error::Error>> {
    let network_weights: FullNetworkWeights = serde_json::from_str(json)?;

    // check that the arrays are the right size
    let neurons = if let Some(factoriser_weight) = &network_weights.factoriser_weight {
        // if we have a factoriser, it gives us the correct number of neurons
        factoriser_weight.len()
    } else {
        // otherwise, there aren't any buckets, so we can safely use the perspective weight
        network_weights.perspective_weight.len()
    };
    println!("neurons: {neurons}");
    let out_size = network_weights.output_weight[0].len();
    println!("out_size: {out_size}");
    if 2 * neurons != out_size {
        return Err(format!(
            "there are {neurons} neurons, but out.weight has {out_size} inputs (should be twice as many)"
        ).into());
    }

    println!("Hope you're using a {neurons}x2 net, because that's what this looks like to me!");

    let buckets = if let Some(factoriser) = &network_weights.factoriser_weight {
        // buckets is the number of times that a factoriser neuron can fit into a perspective neuron
        network_weights.perspective_weight[0].len() / factoriser[0].len()
    } else {
        // if there's no factoriser, there's essentially one bucket.
        1
    };

    if buckets == 1 {
        println!("This net doesn't have any buckets. (or it has one bucket, in which case wtf are you doing?)");
    } else {
        println!("There are {buckets} buckets in this net (we think).");
    }

    // merge the factoriser and perspective weights:
    let merged_net = if let (Some(fft_weight), Some(fft_bias)) = (
        network_weights.factoriser_weight,
        network_weights.factoriser_bias,
    ) {
        let mut perspective_bias = network_weights.perspective_bias;
        assert_eq!(network_weights.perspective_weight.len(), fft_weight.len());
        assert_eq!(perspective_bias.len(), fft_bias.len());
        assert_eq!(
            network_weights.perspective_weight[0].len() / buckets,
            fft_weight[0].len()
        );
        // merge biases
        for (bias_src, bias_dst) in fft_bias.iter().zip(perspective_bias.iter_mut()) {
            *bias_dst += bias_src;
        }
        // merge weights -
        // as far as i can tell, marlinflow generates unbucketed networks like this:
        // [neuron][feature] - this is why we flip it when we quantise it, to get [feature][neuron]
        // the bucketed networks treat the king location as moving all features 768 places to the right, giving us:
        // [neuron][bucket][feature], in effect.
        // we would like to transform this into [feature][neuron][bucket], so that we can quantise it, which will take
        // a bit of jigging around. We can rely on the quantiser to do the right thing to a 2d array, so we can just
        // pretend that the buckets are just extra neurons, like [neurons * buckets][feature]. Then we can flip it
        // to get [feature][neurons * buckets] by using a stride of neurons * buckets.
        let mut reshaped =
            vec![vec![0f64; INPUT_SIZE].into_boxed_slice(); neurons * buckets].into_boxed_slice();
        for (neuron_idx, neuron) in network_weights.perspective_weight.iter().enumerate() {
            // each neuron is a vector of [bucket][feature]
            for (bucket_idx, bucket) in neuron.chunks_exact(INPUT_SIZE).enumerate() {
                // each bucket is a vector of [feature]
                for (feature_idx, feature) in bucket.iter().enumerate() {
                    // each feature is a single value
                    reshaped[neuron_idx + bucket_idx * neurons][feature_idx] = *feature;
                    // add the factoriser weight to the perspective weight
                    reshaped[neuron_idx + bucket_idx * neurons][feature_idx] +=
                        fft_weight[neuron_idx][feature_idx];
                }
            }
        }

        MergedNetworkWeights {
            perspective_weight: reshaped,
            perspective_bias,
            output_weight: network_weights.output_weight,
            output_bias: network_weights.output_bias,
            psqt_weight: network_weights
                .psqt_weight
                .map(|mut x| std::mem::take(&mut x[0])),
        }
    } else {
        MergedNetworkWeights {
            perspective_weight: network_weights.perspective_weight,
            perspective_bias: network_weights.perspective_bias,
            output_weight: network_weights.output_weight,
            output_bias: network_weights.output_bias,
            psqt_weight: network_weights
                .psqt_weight
                .map(|mut x| std::mem::take(&mut x[0])),
        }
    };

    // allocate buffers for the weights and biases
    let mut feature_weights_buf = vec![0i16; neurons * INPUT_SIZE * buckets];
    let mut feature_bias_buf = vec![0i16; neurons];
    let mut output_weights_buf = vec![0i16; out_size];
    let mut output_bias_buf = vec![0i16; 1];
    let mut psqt_buf = vec![0i16; 64 * 12];

    // read the weights and biases into the buffers
    for bucket in 0..buckets {
        let weights = &merged_net.perspective_weight[bucket * neurons..(bucket + 1) * neurons];
        let buffer = &mut feature_weights_buf
            [bucket * neurons * INPUT_SIZE..(bucket + 1) * neurons * INPUT_SIZE];
        quantise_neurons(weights, buffer, neurons, qa, DoTranspose::Yes);
    }
    quantise_biases(&merged_net.perspective_bias, &mut feature_bias_buf, qa);
    quantise_neurons(
        &merged_net.output_weight,
        &mut output_weights_buf,
        out_size,
        qb,
        DoTranspose::No,
    );
    quantise_biases(&merged_net.output_bias, &mut output_bias_buf, qa * qb);
    let psqt_buf = merged_net.psqt_weight.map(|psqt| {
        quantise_psqt(&psqt, &mut psqt_buf, qa * qb);
        psqt_buf
    });

    // return the buffers
    Ok(QuantisedMergedNetwork {
        feature_weights: feature_weights_buf,
        feature_bias: feature_bias_buf,
        output_weights: output_weights_buf,
        output_bias: output_bias_buf,
        psqt_weights: psqt_buf,
        has_buckets: buckets > 1,
        hidden_size: neurons,
    })
}

mod tests {
    #[test]
    fn test_from_json_0030() {
        use crate::convert::QuantisedMergedNetwork;
        let json = std::fs::read_to_string("validation/net0030/viri0030.json").unwrap();
        let QuantisedMergedNetwork {
            feature_weights,
            feature_bias,
            output_weights,
            output_bias,
            ..
        } = crate::convert::from_json(&json, 255, 64).unwrap();
        assert_eq!(feature_weights.len(), 768 * 256);
        assert_eq!(feature_bias.len(), 256);
        assert_eq!(output_weights.len(), 512);
        assert_eq!(output_bias.len(), 1);
        let validation_ft_weight = std::fs::read("validation/net0030/feature_weights.bin").unwrap();
        let validation_ft_bias = std::fs::read("validation/net0030/feature_bias.bin").unwrap();
        let validation_o_weight = std::fs::read("validation/net0030/output_weights.bin").unwrap();
        let validation_o_bias = std::fs::read("validation/net0030/output_bias.bin").unwrap();
        let validation_binary = std::fs::read("validation/net0030/viri0030.bin").unwrap();

        let validation_concat = [
            &validation_ft_weight[..],
            &validation_ft_bias[..],
            &validation_o_weight[..],
            &validation_o_bias[..],
        ]
        .concat();

        assert_eq!(
            validation_binary, validation_concat,
            "validation binary is not as expected"
        );

        let our_concat = [
            &feature_weights[..],
            &feature_bias[..],
            &output_weights[..],
            &output_bias[..],
        ]
        .concat();
        let our_concat = unsafe {
            let inner = our_concat.align_to::<u8>().1;
            inner
        };

        assert_eq!(
            validation_binary, our_concat,
            "our binary is not as expected"
        );
    }

    #[test]
    fn test_from_json_0056() {
        use crate::convert::QuantisedMergedNetwork;
        let json = std::fs::read_to_string("validation/net0056/viri0056.json").unwrap();
        let QuantisedMergedNetwork {
            feature_weights: ft_weights,
            feature_bias: ft_bias,
            output_weights: out_weights,
            output_bias: out_bias,
            ..
        } = crate::convert::from_json(&json, 255, 64).unwrap();
        assert_eq!(ft_weights.len(), 768 * 512);
        assert_eq!(ft_bias.len(), 512);
        assert_eq!(out_weights.len(), 1024);
        assert_eq!(out_bias.len(), 1);
        let validation_ft_weight = std::fs::read("validation/net0056/feature_weights.bin").unwrap();
        let validation_ft_bias = std::fs::read("validation/net0056/feature_bias.bin").unwrap();
        let validation_o_weight = std::fs::read("validation/net0056/output_weights.bin").unwrap();
        let validation_o_bias = std::fs::read("validation/net0056/output_bias.bin").unwrap();
        let validation_binary = std::fs::read("validation/net0056/viri0056.bin").unwrap();

        let validation_concat = [
            &validation_ft_weight[..],
            &validation_ft_bias[..],
            &validation_o_weight[..],
            &validation_o_bias[..],
        ]
        .concat();

        assert_eq!(
            validation_binary, validation_concat,
            "validation binary is not as expected"
        );

        let our_concat = [
            &ft_weights[..],
            &ft_bias[..],
            &out_weights[..],
            &out_bias[..],
        ]
        .concat();
        let our_concat = unsafe {
            let inner = our_concat.align_to::<u8>().1;
            inner
        };

        assert_eq!(
            validation_binary, our_concat,
            "our binary is not as expected"
        );
    }
}
