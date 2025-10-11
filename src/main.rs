mod app;

use anyhow::Result;

fn main() -> Result<()> {
    app::AurionApp::run()
}
