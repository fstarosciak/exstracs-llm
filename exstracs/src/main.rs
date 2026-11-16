use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};
use exstracs::model::default_rule_specificity_limit;
use exstracs::multiplexer::multiplexer_table;
use exstracs::rules::validate;
use exstracs::{Config, Dataset, ExSTraCS, LoadOptions, RuleFile, Selection, Snapshot, TrainStats};
use serde_json::json;

const CURVE_HEADER: [&str; 10] = [
    "iteration",
    "tracked_accuracy",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_coverage",
    "macro_size",
    "micro_size",
    "mean_specificity",
    "seeded_alive",
    "seed_descendants",
];

#[derive(Parser, Debug)]
#[command(
    name = "exstracs",
    version,
    about = "ExSTraCS supervised learning classifier system with rule-population seeding"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    #[command(
        about = "Train a model, optionally seeded with external rules, and report a learning curve"
    )]
    Train(Box<TrainArgs>),
    #[command(about = "Evaluate a saved population snapshot on a labelled CSV file")]
    Predict(PredictArgs),
    #[command(about = "Validate a seed rule file against a training data schema")]
    ValidateRules(ValidateRulesArgs),
    #[command(about = "Generate a multiplexer benchmark CSV file")]
    GenMux(GenMuxArgs),
}

#[derive(Args, Debug)]
struct DataArgs {
    #[arg(long, value_name = "PATH", help = "Training data CSV file")]
    train: PathBuf,
    #[arg(
        long,
        value_name = "NAME",
        default_value = "class",
        help = "Name of the class column"
    )]
    class_column: String,
    #[arg(
        long,
        value_name = "N",
        default_value_t = 10,
        help = "Attributes with at most this many distinct values are treated as discrete"
    )]
    discrete_limit: usize,
    #[arg(
        long,
        value_name = "A,B",
        value_delimiter = ',',
        help = "Attributes forced to be discrete"
    )]
    discrete: Vec<String>,
    #[arg(
        long,
        value_name = "A,B",
        value_delimiter = ',',
        help = "Attributes forced to be continuous"
    )]
    continuous: Vec<String>,
    #[arg(
        long,
        value_name = "A,B",
        value_delimiter = ',',
        help = "Columns to ignore"
    )]
    ignore: Vec<String>,
}

impl DataArgs {
    fn load_options(&self) -> LoadOptions {
        LoadOptions {
            class_column: self.class_column.clone(),
            discrete_limit: self.discrete_limit,
            discrete: self.discrete.clone(),
            continuous: self.continuous.clone(),
            ignore: self.ignore.clone(),
            ..LoadOptions::default()
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SelectionArg {
    Tournament,
    Roulette,
}

#[derive(Args, Debug)]
#[command(next_help_heading = "Config overrides")]
struct ConfigOverrides {
    #[arg(long, value_name = "N", help = "Maximum micro population size")]
    population_size: Option<u64>,
    #[arg(long, value_name = "N", help = "Fitness exponent")]
    nu: Option<u32>,
    #[arg(long, value_name = "P", help = "Crossover probability")]
    chi: Option<f64>,
    #[arg(long, value_name = "P", help = "Mutation probability")]
    mu: Option<f64>,
    #[arg(long, value_name = "X", help = "GA invocation threshold")]
    theta_ga: Option<f64>,
    #[arg(long, value_name = "N", help = "Deletion experience threshold")]
    theta_del: Option<u32>,
    #[arg(long, value_name = "N", help = "Subsumption experience threshold")]
    theta_sub: Option<u32>,
    #[arg(long, value_name = "P", help = "Subsumption accuracy threshold")]
    acc_sub: Option<f64>,
    #[arg(long, value_name = "X", help = "Learning rate")]
    beta: Option<f64>,
    #[arg(long, value_name = "P", help = "Deletion fitness fraction")]
    delta: Option<f64>,
    #[arg(long, value_name = "X", help = "Initial fitness of new classifiers")]
    init_fitness: Option<f64>,
    #[arg(long, value_name = "P", help = "Fitness reduction for offspring")]
    fitness_reduction: Option<f64>,
    #[arg(long, value_name = "BOOL", help = "Enable GA subsumption")]
    ga_subsumption: Option<bool>,
    #[arg(long, value_name = "BOOL", help = "Enable correct set subsumption")]
    correct_set_subsumption: Option<bool>,
    #[arg(long, value_enum, help = "Parent selection method")]
    selection: Option<SelectionArg>,
    #[arg(
        long,
        value_name = "P",
        help = "Tournament size as a fraction of the correct set"
    )]
    theta_sel: Option<f64>,
    #[arg(long, value_name = "N", help = "Rule specificity limit")]
    rsl: Option<usize>,
    #[arg(long, value_name = "N", help = "Random seed")]
    seed: Option<u64>,
    #[arg(long, help = "Keep seeded rule conditions fixed")]
    fixed_conditions: bool,
}

impl ConfigOverrides {
    fn apply(&self, config: &mut Config) {
        config.population_size = self.population_size.unwrap_or(config.population_size);
        config.nu = self.nu.unwrap_or(config.nu);
        config.chi = self.chi.unwrap_or(config.chi);
        config.mu = self.mu.unwrap_or(config.mu);
        config.theta_ga = self.theta_ga.unwrap_or(config.theta_ga);
        config.theta_del = self.theta_del.unwrap_or(config.theta_del);
        config.theta_sub = self.theta_sub.unwrap_or(config.theta_sub);
        config.acc_sub = self.acc_sub.unwrap_or(config.acc_sub);
        config.beta = self.beta.unwrap_or(config.beta);
        config.delta = self.delta.unwrap_or(config.delta);
        config.init_fitness = self.init_fitness.unwrap_or(config.init_fitness);
        config.fitness_reduction = self.fitness_reduction.unwrap_or(config.fitness_reduction);
        config.do_ga_subsumption = self.ga_subsumption.unwrap_or(config.do_ga_subsumption);
        config.do_correct_set_subsumption = self
            .correct_set_subsumption
            .unwrap_or(config.do_correct_set_subsumption);
        if let Some(selection) = self.selection {
            config.selection = match selection {
                SelectionArg::Tournament => Selection::Tournament,
                SelectionArg::Roulette => Selection::Roulette,
            };
        }
        config.theta_sel = self.theta_sel.unwrap_or(config.theta_sel);
        config.rule_specificity_limit = self.rsl.or(config.rule_specificity_limit);
        config.seed = self.seed.unwrap_or(config.seed);
        config.fixed_conditions |= self.fixed_conditions;
    }
}

#[derive(Args, Debug)]
struct TrainArgs {
    #[command(flatten)]
    data: DataArgs,
    #[arg(long, value_name = "PATH", help = "Test data CSV file")]
    test: Option<PathBuf>,
    #[arg(long, value_name = "PATH", help = "Seed rule file (JSON)")]
    rules: Option<PathBuf>,
    #[arg(
        long,
        value_name = "N",
        default_value_t = 1,
        value_parser = clap::value_parser!(u32).range(1..),
        help = "Numerosity of each seeded rule"
    )]
    seed_numerosity: u32,
    #[arg(
        long,
        value_name = "N",
        default_value_t = 100000,
        help = "Number of training iterations"
    )]
    iterations: u64,
    #[arg(
        long,
        value_name = "N",
        default_value_t = 1000,
        help = "Iterations between learning curve rows (0 for a single step)"
    )]
    report_every: u64,
    #[arg(long, value_name = "PATH", help = "Config file (JSON)")]
    config: Option<PathBuf>,
    #[arg(long, value_name = "DIR", help = "Output directory")]
    out: Option<PathBuf>,
    #[command(flatten)]
    overrides: ConfigOverrides,
}

#[derive(Args, Debug)]
struct PredictArgs {
    #[arg(long, value_name = "PATH", help = "Population snapshot (JSON)")]
    model: PathBuf,
    #[arg(long, value_name = "PATH", help = "Labelled data CSV file")]
    data: PathBuf,
    #[arg(
        long,
        value_name = "A,B",
        value_delimiter = ',',
        help = "Columns to ignore"
    )]
    ignore: Vec<String>,
    #[arg(long, value_name = "PATH", help = "Per-row predictions CSV file")]
    out: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct ValidateRulesArgs {
    #[command(flatten)]
    data: DataArgs,
    #[arg(long, value_name = "PATH", help = "Seed rule file (JSON)")]
    rules: PathBuf,
    #[arg(
        long,
        value_name = "N",
        help = "Rule specificity limit (default derived from the training data)"
    )]
    rsl: Option<usize>,
}

#[derive(Args, Debug)]
struct GenMuxArgs {
    #[arg(long, value_name = "K", help = "Number of address bits")]
    address_bits: usize,
    #[arg(long, value_name = "PATH", help = "Output CSV file")]
    out: PathBuf,
    #[arg(
        long,
        value_name = "N",
        help = "Number of random samples (default: full truth table)"
    )]
    samples: Option<usize>,
    #[arg(long, value_name = "N", default_value_t = 42, help = "Random seed")]
    seed: u64,
}

fn cell(value: Option<f64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn curve_row(model: &ExSTraCS, tracked: Option<f64>, test: Option<&Dataset>) -> Vec<String> {
    let metrics = test.map(|data| model.evaluate(data));
    let population = model.population_stats();
    vec![
        model.iteration().to_string(),
        cell(tracked),
        cell(metrics.map(|metrics| metrics.accuracy)),
        cell(metrics.map(|metrics| metrics.balanced_accuracy)),
        cell(metrics.map(|metrics| metrics.coverage)),
        population.macro_size.to_string(),
        population.micro_size.to_string(),
        population.mean_specificity.to_string(),
        population.seeded_alive.to_string(),
        population.seed_descendants.to_string(),
    ]
}

fn train(args: TrainArgs) -> Result<(), Box<dyn Error>> {
    let options = args.data.load_options();
    let train = Dataset::from_csv(&args.data.train, &options)?;
    let test = match &args.test {
        Some(path) => Some(Dataset::from_csv_with_schema(
            path,
            &train.schema,
            &options,
        )?),
        None => None,
    };
    let mut config: Config = match &args.config {
        Some(path) => serde_json::from_str(&std::fs::read_to_string(path)?)?,
        None => Config::default(),
    };
    args.overrides.apply(&mut config);
    let mut model = ExSTraCS::new(config, &train)?;

    let mut seeding = None;
    let mut report = None;
    if let Some(path) = &args.rules {
        let file = RuleFile::load(path)?;
        let validated = validate(&file, model.schema(), model.rule_specificity_limit());
        let inserted = model.seed(&validated.rules, args.seed_numerosity);
        let total = validated.report.total;
        let accepted = validated.report.accepted;
        let rejected = validated.report.rejected.len();
        eprintln!("seeded {inserted} rules ({accepted} accepted, {rejected} rejected of {total})");
        seeding = Some(json!({
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "inserted": inserted,
        }));
        report = Some(validated.report);
    }

    let step = if args.report_every == 0 {
        args.iterations
    } else {
        args.report_every
    };
    let mut remaining = args.iterations;
    let mut total = TrainStats::default();
    let mut elapsed = Duration::ZERO;
    let mut curve = vec![curve_row(&model, None, test.as_ref())];
    while remaining > 0 {
        let n = step.min(remaining);
        let start = Instant::now();
        let stats = model.train(&train, n)?;
        elapsed += start.elapsed();
        remaining -= n;
        total.iterations += stats.iterations;
        total.correct += stats.correct;
        total.covering += stats.covering;
        total.ga_runs += stats.ga_runs;
        let tracked =
            (stats.iterations > 0).then(|| stats.correct as f64 / stats.iterations as f64);
        curve.push(curve_row(&model, tracked, test.as_ref()));
    }

    let summary = json!({
        "train_file": args.data.train.display().to_string(),
        "test_file": args.test.as_ref().map(|path| path.display().to_string()),
        "rules_file": args.rules.as_ref().map(|path| path.display().to_string()),
        "n_train": train.len(),
        "n_test": test.as_ref().map(Dataset::len),
        "n_attributes": train.schema.n_attributes(),
        "classes": train.schema.classes,
        "rule_specificity_limit": model.rule_specificity_limit(),
        "iterations": total.iterations,
        "runtime_seconds": elapsed.as_secs_f64(),
        "config": model.config(),
        "train": model.evaluate(&train),
        "test": test.as_ref().map(|data| model.evaluate(data)),
        "population": model.population_stats(),
        "training": total,
        "seeding": seeding,
    });
    let summary_text = serde_json::to_string_pretty(&summary)?;
    println!("{summary_text}");

    if let Some(dir) = &args.out {
        std::fs::create_dir_all(dir)?;
        model.snapshot().save(&dir.join("population.json"))?;
        let mut writer = csv::Writer::from_path(dir.join("curve.csv"))?;
        writer.write_record(CURVE_HEADER)?;
        for row in &curve {
            writer.write_record(row)?;
        }
        writer.flush()?;
        std::fs::write(dir.join("summary.json"), summary_text + "\n")?;
        if let Some(report) = &report {
            std::fs::write(
                dir.join("rules_report.json"),
                serde_json::to_string_pretty(report)? + "\n",
            )?;
        }
        eprintln!("wrote {}", dir.display());
    }
    Ok(())
}

fn predict(args: PredictArgs) -> Result<(), Box<dyn Error>> {
    let model = ExSTraCS::from_snapshot(Snapshot::load(&args.model)?)?;
    let options = LoadOptions {
        ignore: args.ignore,
        ..LoadOptions::default()
    };
    let data = Dataset::from_csv_with_schema(&args.data, model.schema(), &options)?;
    println!("{}", serde_json::to_string_pretty(&model.evaluate(&data))?);
    if let Some(path) = &args.out {
        let classes = &model.schema().classes;
        let class_name = |index: u32| classes.get(index as usize).cloned().unwrap_or_default();
        let mut writer = csv::Writer::from_path(path)?;
        writer.write_record(["row", "actual", "predicted", "covered"])?;
        for (row, (label, prediction)) in
            data.labels.iter().zip(model.predict_all(&data)).enumerate()
        {
            writer.write_record([
                row.to_string(),
                class_name(*label),
                class_name(prediction.class),
                prediction.covered.to_string(),
            ])?;
        }
        writer.flush()?;
    }
    Ok(())
}

fn validate_rules(args: ValidateRulesArgs) -> Result<(), Box<dyn Error>> {
    let train = Dataset::from_csv(&args.data.train, &args.data.load_options())?;
    let rsl = args
        .rsl
        .unwrap_or_else(|| default_rule_specificity_limit(&train));
    let file = RuleFile::load(&args.rules)?;
    let validated = validate(&file, &train.schema, rsl);
    println!("{}", serde_json::to_string_pretty(&validated.report)?);
    Ok(())
}

fn gen_mux(args: GenMuxArgs) -> Result<(), Box<dyn Error>> {
    let table = multiplexer_table(args.address_bits, args.samples, args.seed)?;
    table.write_csv(&args.out)?;
    eprintln!("wrote {} rows to {}", table.rows.len(), args.out.display());
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    match Cli::parse().command {
        Command::Train(args) => train(*args),
        Command::Predict(args) => predict(args),
        Command::ValidateRules(args) => validate_rules(args),
        Command::GenMux(args) => gen_mux(args),
    }
}
