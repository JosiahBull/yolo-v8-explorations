use std::{fs::read_dir, path::{Path, PathBuf}, sync::atomic::{AtomicUsize, Ordering}};

use image::{ImageBuffer, Rgb, Pixel};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

const IGNORED_REGIONS: &[&(Coordinate, Coordinate)] = &[
    &(Coordinate { x: 0, y: 0 }, Coordinate { x: 339, y: 346}), // friendly team members
    &(Coordinate { x: 691, y: 0 }, Coordinate { x: 1869, y: 100 }), // game status
    &(Coordinate { x: 2178, y: 0 }, Coordinate { x: 2560, y: 352 }), // enemy team members

    &(Coordinate { x: 0, y: 1216 }, Coordinate { x: 564, y: 1440 }), // health bar
    &(Coordinate { x: 2079, y: 960 }, Coordinate { x: 2560, y: 1440 }), // minimap
];

const TARGET_COLORS: [u8; 3] = [222, 35, 28]; // enemy health bar

#[derive(Debug)]
struct Coordinate {
    x: u32,
    y: u32,
}

#[derive(Debug)]
enum FrameState {
    Uncategorised,
    NoTargets,
    Targets,
    HitBoxes(Vec<(Coordinate, Coordinate)>),
    Unsure(Vec<(Coordinate, Coordinate)>),
}

#[derive(Debug)]
struct Frame {
    state: FrameState,
    image: Option<ImageBuffer<Rgb<u8>, Vec<u8>>>,
    file_path: Option<PathBuf>,
}

fn detect_targets(frame: &mut Frame) {
    // loop through each pixel in the image - if within 10% of the target color, mark this frame as a target
    // if not, mark it as no targets

    for (x, y, pixel) in frame.image.as_mut().unwrap().enumerate_pixels_mut() {
        // skip pixel if it is in an ignored region
        let mut ignore = false;
        for region in IGNORED_REGIONS {
            if x >= region.0.x && x <= region.1.x && y >= region.0.y && y <= region.1.y {
                ignore = true;
                break;
            }
        }
        if ignore {
            continue;
        }

        let pixel_colors: [u8; 3] = pixel.channels().try_into().unwrap();
        let r = pixel_colors[0];
        let g = pixel_colors[1];
        let b = pixel_colors[2];

        // if the pixel is within 10% of the target color, mark this frame as a target
        if (r as f32 - TARGET_COLORS[0] as f32).abs() < 0.1 * TARGET_COLORS[0] as f32
            && (g as f32 - TARGET_COLORS[1] as f32).abs() < 0.1 * TARGET_COLORS[1] as f32
            && (b as f32 - TARGET_COLORS[2] as f32).abs() < 0.1 * TARGET_COLORS[2] as f32
        {
            frame.state = FrameState::Targets;

            // println!("Found target at {}, {}", x, y);

            return;
        }
    }

    frame.state = FrameState::NoTargets;
}

fn recursively_find_frames(target_dir: &Path) -> Vec<Frame> {
    let mut frames: Vec<Frame> = Vec::new();

    for entry in read_dir(target_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_dir() {
            frames.append(&mut recursively_find_frames(&path));
        } else {
            let frame = Frame {
                state: FrameState::Uncategorised,
                image: None,
                file_path: Some(path),
            };

            frames.push(frame);
        }
    }

    frames
}

fn process_frames(frames: &mut Vec<Frame>) {
    let counter = AtomicUsize::new(0);
    let total_frames = frames.len();
    frames.par_iter_mut().for_each(|frame| {
        let image = image::open(frame.file_path.as_ref().unwrap()).unwrap().to_rgb8();
        frame.image = Some(image);
        detect_targets(frame);
        frame.image = None;

        println!("Processed frame {}/{}", counter.fetch_add(1, Ordering::Relaxed), total_frames);
    });
}

fn find_frames_with_targets(target_dir: &str, out_dir: &str) {
    let mut frames = recursively_find_frames(Path::new(target_dir));
    process_frames(&mut frames);

    // save the frames to disk in the output directory using folders for each state
    for frame in frames {
        let state_dir = match frame.state {
            FrameState::Uncategorised => "uncategorised",
            FrameState::NoTargets => "no_targets",
            FrameState::Targets => "targets",
            FrameState::HitBoxes(_) => "hitboxes",
            FrameState::Unsure(_) => "unsure",
        };

        let parent_path = frame.file_path.as_ref().unwrap().parent().unwrap().file_name().unwrap();
        let out_path = Path::new(out_dir).join(parent_path).join(state_dir).join(frame.file_path.as_ref().unwrap().file_name().unwrap());

        // create the directory if it doesn't exist
        std::fs::create_dir_all(out_path.parent().unwrap()).unwrap();
        std::fs::copy(
            frame.file_path.as_ref().unwrap(),
            out_path,
        ).unwrap();
    }
}

fn main() {
    // The goal of this program is to try and find the hitbox of an enemy by scanning an image

    const TARGET_DIR: &str = "./data/frames/";
    const OUT_DIR: &str = "./data/processed_frames/";

    find_frames_with_targets(TARGET_DIR, OUT_DIR);


}
