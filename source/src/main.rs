mod gpu;
mod weights;
mod model;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::{io::{self, BufRead, Write}, path::PathBuf,
    sync::{mpsc, atomic::Ordering}};

fn run()->Result<()> {
    let mut args=std::env::args().skip(1);
    let mut path=None;
    while let Some(arg)=args.next(){match arg.as_str(){"--model"=>path=args.next().map(PathBuf::from),_=>bail!("unknown argument {arg}")}}
    let path=path.context("--model MODEL_DIR required")?;
    let mut model=model::Model::load(&path)?;
    let cancelled=model.cancelled.clone();
    let (sender,receiver)=mpsc::sync_channel(1);
    std::thread::spawn(move||{
        for line in io::stdin().lock().lines() {
            let failed=line.is_err();
            if sender.send(line).is_err()||failed {break;}
        }
        // Disconnect stops inference after the currently submitted GPU work.
        cancelled.store(true,Ordering::Relaxed);
    });
    let mut stdout=io::BufWriter::new(io::stdout().lock());
    for line in receiver {
        if model.cancelled.load(Ordering::Relaxed) {break;}
        let line=line?;
        let req:Value=serde_json::from_str(&line)?;
        let id=req.get("id").cloned().unwrap_or(Value::Null);
        let result=(||->Result<Value>{
            match req["op"].as_str().context("missing op")? {
                "hello"=>Ok(json!({"protocol":1})),
                "generate"=>{
                    let prompt=req["prompt"].as_array().context("missing prompt")?.iter()
                        .map(|x|x.as_u64().and_then(|x|u32::try_from(x).ok()).context("invalid token")).collect::<Result<Vec<_>>>()?;
                    let count=req["max_tokens"].as_u64().context("missing max_tokens")? as usize;
                    let reset=req["reset"].as_bool().context("missing reset")?;
                    let generated=model.generate(&prompt,count,reset);
                    Ok(json!({"tokens":generated?}))
                }
                op=>bail!("unknown operation {op}"),
            }
        })();
        if model.cancelled.load(Ordering::Relaxed) {break;}
        let mut reply=match result {Ok(v)=>v,Err(e)=>json!({"ok":false,"error":format!("{e:#}")})};
        if reply.get("ok").is_none(){reply["ok"]=json!(true);}
        reply["id"]=id;
        serde_json::to_writer(&mut stdout,&reply)?;stdout.write_all(b"\n")?;stdout.flush()?;
    }
    Ok(())
}
fn main(){if let Err(e)=objc2::rc::autoreleasepool(|_|run()){eprintln!("engine: {e:#}");std::process::exit(1);}}
